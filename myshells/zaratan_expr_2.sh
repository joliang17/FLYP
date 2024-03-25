#!/bin/bash
#SBATCH --job-name=flyp_loss_base_vitl
#SBATCH --account=tianyi-prj-cmsc
#SBATCH --time=24:00:00
#SBATCH --gpus=a100:2
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=128
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=6

. ~/.bashrc
export SLURM_EXPORT_ENV=ALL

# Section to output information identifying the job, etc.
echo "Slurm job ${SLURM_JOBID} running on ${hostname}"
echo "To run on ${SLURM_NTASKS} CPU cores across ${SLURM_JOB_NUM_NODES} nodes"
echo "All nodes: ${SLURM_JOB_NODELIST} ${date} ${pwd}"

current_dir=$(pwd)
if [[ "$current_dir" != *"/nexus-scratch/"* ]]; then
    # on zaratan cluster

    conda activate diffu
    scratch_root="/home/yliang17/scratch.tianyi-prj/Research/"

    # location of original imgs
    root_folder="${scratch_root}/gene_diffcls"
    # location of cache files
    cache_folder="${scratch_root}/cache"

    # location of generated images
    shell_root="/afs/shell.umd.edu/project/tianyi-prj/user/yliang17/Research"
    shell_folder="${shell_root}/gene_diffcls"
    IMG_FOLDER="${shell_folder}/data/train_new"
    
    echo "${current_dir}"
    echo "On zaratan node"
else
    # on uniacs cluster

    source /fs/nexus-scratch/yliang17/miniconda3/bin/activate diffu
    scratch_root="/fs/nexus-scratch/yliang17/Research"

    # location of original imgs
    root_folder=".."
    # location of cache files
    cache_folder="${scratch_root}/cache"

    # location of generated images
    IMG_FOLDER="${root_folder}/data/train_new"
fi


# location of generated train.csv / curriculum.csv / used pkl ..
META_FOLDER="${root_folder}/data/metadata"
SAVED_FOLDER="${META_FOLDER}/clip_progress_difficult_2022_5_onlyguid_zrt/"

#python datacreation_scripts/iwildcam.py --save_folder="${SAVED_FOLDER}" --input_folder="${IMG_FOLDER}" --curriculum --gene_constr="${META_FOLDER}/used_imgid/used_imgid_v5.pkl"

# uniform dataset + guid >= 50
python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=128 --model=ViT-L-14 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save="./checkpoints/" --data-location="../data/iwildcam/" --ft_data="../data/iwildcam/iwildcam_v2.0/train.csv" --cache_folder="${cache_folder}" --csv-img-key filepath --csv-caption-key title --workers=4 --exp_name="vit_L" --baseline --slurm_job_id=$SLURM_JOB_ID --debug


ECODE=$?
	
echo "Job finished with exit code ${ECODE} ${date}"

# Exit with the cached exit code
exit $ECODE
