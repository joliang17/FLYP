#!/bin/bash

#SBATCH --job-name=flyp_loss_curri_prog_v33_2
#SBATCH --output=flyp_loss_curri_prog_v33_2.out.%j
#SBATCH --error=flyp_loss_curri_prog_v33_2.out.%j
#SBATCH --time=24:00:00
#SBATCH --account=scavenger 
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G


# cd ../..
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate diffu


TRAIN_FOLDER="../data/train_new"
TEST_FOLDER="../data/test"
SAVED_FOLDER="../data/metadata/clip_newcurri/"

python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=200 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location="./datasets/data/" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_test="${SAVED_FOLDER}curriculum.csv" --csv-img-key filepath --csv-caption-key title --exp_name=flyp_loss_curri_prog_v33_2 --curriculum --curriculum_epoch=10 --scheduler=default --workers=4 --progress --explore


