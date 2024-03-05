#!/bin/bash

#SBATCH --job-name=flyp_loss_v52
#SBATCH --output=flyp_loss_v52.out.%j
#SBATCH --error=flyp_loss_v52.out.%j
#SBATCH --time=48:00:00
#SBATCH --account=cml-zhou
#SBATCH --partition=cml-dpart
#SBATCH --qos=cml-medium
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G




# checking gpu status
nvidia-smi

# cd ../..
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate diffu


TRAIN_FOLDER="../data/train_new"
TEST_FOLDER="../data/test"
SAVED_FOLDER="../data/metadata/clip_progress_train/"

#  python datacreation_scripts/iwildcam.py --mode="train" --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER} --curriculum --total_train
# python datacreation_scripts/iwildcam.py --mode="curriculum" --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER} --curriculum --total_train
# python datacreation_scripts/iwildcam.py --mode="test" --save_folder=${SAVED_FOLDER} --input_folder=${TEST_FOLDER} --curriculum

# curriculum tau version
python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=300 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location="./datasets/data/" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_test="${SAVED_FOLDER}train.csv" --csv-img-key filepath --csv-caption-key title --exp_name="flyp_loss_v52" --curriculum --curriculum_epoch=10 --progress_eval --progress_train --scheduler=default --workers=4 --progress_metric="Acc" --progress --explore --slurm_job_id=$SLURM_JOB_ID --cont_finetune --explore_fixguid

