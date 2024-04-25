#!/bin/bash
#SBATCH --job-name=v772_p
#SBATCH --account=tianyi-prj-cmsc
#SBATCH --time=24:00:00
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=10000
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

    # location of cache files
    cache_folder="${scratch_root}/cache"

    echo "On zaratan node"
else
    # on uniacs cluster

    source /fs/nexus-scratch/yliang17/miniconda3/bin/activate diffu
    scratch_root="/fs/nexus-scratch/yliang17/Research"

    # location of cache files
    cache_folder="${scratch_root}/cache"
fi


# location of original imgs
root_folder=".."
# location of generated train.csv / curriculum.csv / used pkl ..
IMG_FOLDER="${root_folder}/data/train_new"
META_FOLDER="${root_folder}/data/metadata"
SAVED_FOLDER="${META_FOLDER}/difficult_2_sample/"


#python datacreation_scripts/iwildcam.py --save_folder="${SAVED_FOLDER}" --input_folder="${IMG_FOLDER}" --curriculum --gene_constr="${META_FOLDER}/used_imgid/used_imgid_v2.pkl" --data_folder="${root_folder}" --sample_guid


# uniform dataset + guid >= 50
python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=150 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save="./checkpoints/" --data-location="${root_folder}/data/iwildcam/" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_test="${SAVED_FOLDER}curriculum.csv"  --cache_folder="${cache_folder}" --csv-img-key filepath --csv-caption-key title --workers=4 --exp_name="flyp_loss_v772_p" --curriculum --curriculum_epoch=15 --progress_sample --partial_update --merge_ori --scheduler=default --slurm_job_id=$SLURM_JOB_ID --debug


ECODE=$?

echo "Job finished with exit code ${ECODE} ${date}"

# Exit with the cached exit code
exit $ECODE
