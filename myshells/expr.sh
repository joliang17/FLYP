#!/bin/bash

#SBATCH --job-name=flyp_loss_curri_strength
#SBATCH --output=flyp_loss_curri_strength.out.%j
#SBATCH --error=flyp_loss_curri_strength.out.%j
#SBATCH --time=48:00:00
#SBATCH --account=scavenger 
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G


# cd ../..
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate diffu


TRAIN_FOLDER="../data/train_new"
TEST_FOLDER="../data/test"
SAVED_FOLDER="../data/metadata/clip_newcurri_v2/"

# limits=( 0 10 30 50 80 90 100 )
# limits=( 0 )

python datacreation_scripts/iwildcam.py --mode="train" --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER} --curriculum --total_train
# python datacreation_scripts/iwildcam.py --mode="curriculum" --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER} --curriculum --total_train

# python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=256 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location="./datasets/data/" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_test="${SAVED_FOLDER}curriculum.csv" --csv-img-key filepath --csv-caption-key title --exp_name=flyp_loss_curri_progress --progress_eval --curriculum --curriculum_epoch=5 --scheduler=drestart --debug --cont_finetune

# for i in "${limits[@]}"
# do
# 	echo "$i"
#     python src/main.py --train-dataset=IWildCamIDVal --epochs=3 --lr=1e-5 --wd=0.2 --batch-size=64 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location="./datasets/data/" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_test="${SAVED_FOLDER}curriculum.csv" --csv-img-key filepath --csv-caption-key title --exp_name="flyp_loss_curri_progress_${i}_3" --progress_eval --strength=$i --workers=4 --ma_progress --datalimit=60000
# done


