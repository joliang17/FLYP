#!/bin/bash
#SBATCH --job-name=flyp_loss_base_vitl
#SBATCH --account=tianyi-prj-cmsc
#SBATCH --time=2:00:00
#SBATCH --gpus=a100:2
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=128
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=6

. ~/.bashrc
export SLURM_EXPORT_ENV=ALL
# module purge
# module load python

# Section to output information identifying the job, etc.
echo "Slurm job ${SLURM_JOBID} running on ${hostname}"
echo "To run on ${SLURM_NTASKS} CPU cores across ${SLURM_JOB_NUM_NODES} nodes"
echo "All nodes: ${SLURM_JOB_NODELIST} ${date} ${pwd}"
# echo "Loaded modules are:"
# module list

# checking gpu status
# nvidia-smi

current_dir=$(pwd)
if [[ "$current_dir" == *"zt1/project/tianyi-prj"* ]]; then
    # on zaratan cluster

    conda activate flyp
    root_folder="/scratch/zt1/project/tianyi-prj/user/yliang17/Research"
    saved_ckpt="."
    saved_data="${root_folder}/gene_diffcls"

    echo "On zaratan node"
else
    # on uniacs cluster

    source /fs/nexus-scratch/yliang17/miniconda3/bin/activate diffu
    root_folder="/fs/nexus-scratch/yliang17/Research"
    saved_ckpt="."
    saved_data="${root_folder}/diffusion/gene_diffcls"
fi

python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=128 --model=ViT-L-14 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save="${saved_ckpt}/checkpoints/" --data-location="${saved_data}/data/iwildcam/" --ft_data="${saved_data}/data/iwildcam/iwildcam_v2.0/train.csv" --csv-img-key filepath --csv-caption-key title --exp_name=flyp_loss_base_vitl --workers=4 --baseline  --slurm_job_id=$SLURM_JOB_ID --cache_folder="${root_folder}/cache"

ECODE=$?
	
echo "Job finished with exit code ${ECODE} ${date}"

# Exit with the cached exit code
exit $ECODE
