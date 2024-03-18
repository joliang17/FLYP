#!/bin/bash
#SBATCH --job-name=flyp_loss_base_vitl
#SBATCH --ntasks=1
#SBATCH --account=tianyi-prj-cmsc
#SBATCH --time=72:00:00
#SBATCH --gpus=a100:2
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=128
#SBATCH --cpus-per-task=6

. ~/.bashrc
export SLURM_EXPORT_ENV=ALL
module purge
module load python

# Section to output information identifying the job, etc.
echo "Slurm job ${SLURM_JOBID} running on ${hostname}"
echo "To run on ${SLURM_NTASKS} CPU cores across ${SLURM_JOB_NUM_NODES} nodes"
echo "All nodes: ${SLURM_JOB_NODELIST} ${date} ${pwd}"
echo "Loaded modules are:"
module list

# checking gpu status
# nvidia-smi

current_dir=$(pwd)
if [[ "$current_dir" == *"/home/yliang17/Research"* ]]; then
    # on zaratan cluster

    source ~/miniconda3/bin/activate diffu
    root_folder="/afs/shell.umd.edu/project/tianyi-prj/user/yliang17"
    saved_ckpt="${root_folder}/Research/gene_diffcls/FLYP"
    saved_data="${root_folder}/Research/gene_diffcls"
else
    # on uniacs cluster
    source /fs/nexus-scratch/yliang17/miniconda3/bin/activate diffu
    root_folder="/fs/nexus-scratch/yliang17/"
    saved_ckpt="."
    saved_data="${root_folder}/Research/diffusion/gene_diffcls"
fi

TRAIN_FOLDER="${saved_data}/data/train_new"
TEST_FOLDER="${saved_data}/data/test"
SAVED_FOLDER="${saved_data}/data/metadata/clip_original/"

#  python datacreation_scripts/iwildcam.py --mode="train" --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER} --data_folder="${saved_data}" --curriculum --total_train 
# python datacreation_scripts/iwildcam.py --mode="curriculum" --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER} --data_folder="${saved_data}" --curriculum --total_train 

python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=128 --model=ViT-L-14 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save="${saved_ckpt}/checkpoints/" --data-location="${saved_data}/datasets/data/" --ft_data="${saved_data}/datasets/csv/iwildcam_v2.0/train.csv" --csv-img-key filepath --csv-caption-key title --exp_name=flyp_loss_base_vitl --workers=4 --baseline  --slurm_job_id=$SLURM_JOB_ID > job.out 2>&1

ECODE=$?
	
# cp job.out ${SLURM_SUBMIT_DIR}
echo "Job finished with exit code ${ECODE} ${date}"
# Exit with the cached exit code
exit $ECODE
