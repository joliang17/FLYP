#!/bin/bash

#SBATCH --job-name=flyp_loss_curri_strength_1
#SBATCH --output=flyp_loss_curri_strength_1.out.%j
#SBATCH --error=flyp_loss_curri_strength_1.out.%j
#SBATCH --time=48:00:00
#SBATCH --account=cml-zhou
#SBATCH --partition=cml-dpart
#SBATCH --qos=cml-medium
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G



# cd ../..
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate diffu


TRAIN_FOLDER="../data/train_new"
TEST_FOLDER="../data/test"
SAVED_FOLDER="../data/metadata/clip_newcurri/"

python src/main.py --train-dataset=IWildCamIDVal --epochs=2 --lr=1e-5 --wd=0.2 --batch-size=128 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location="./datasets/data/" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_test="${SAVED_FOLDER}curriculum.csv" --csv-img-key filepath --csv-caption-key title --exp_name=progress_validation_1 --progress_eval --curriculum --curriculum_epoch=5 --scheduler=default --progress_validation


