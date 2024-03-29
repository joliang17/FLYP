#!/bin/bash

#SBATCH --job-name=v695
#SBATCH --output=v695.out.%j
#SBATCH --error=v695.out.%j
#SBATCH --time=12:00:00
#SBATCH --account=scavenger 
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G



# checking gpu status
nvidia-smi

# cd ../..
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate diffu

TRAIN_FOLDER="../data/train_new"
TEST_FOLDER="../data/test"
SAVED_FOLDER="../data/metadata/clip_progress_difficult_2022_5_analysis/"


# Use avg of prob diff
python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=100 --model=ViT-B/16 --eval-datasets=IWildCamOOD --template=iwildcam_template --save=./checkpoints/ --data-location="../data/iwildcam/" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_test="${SAVED_FOLDER}train.csv" --csv-img-key filepath --csv-caption-key title --exp_name="noisy_label_loss" --scheduler=default --baseline --slurm_job_id=$SLURM_JOB_ID --debug --workers=4 --cont_finetune

