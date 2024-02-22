#!/bin/bash

#SBATCH --job-name=flyp_loss_prog_train
#SBATCH --output=flyp_loss_prog_train.out.%j
#SBATCH --error=flyp_loss_prog_train.out.%j
#SBATCH --time=24:00:00
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
SAVED_FOLDER="../data/metadata/clip_progress_train/"

#  python datacreation_scripts/iwildcam.py --mode="train" --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER} --curriculum --total_train
# python datacreation_scripts/iwildcam.py --mode="curriculum" --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER} --curriculum --total_train
# python datacreation_scripts/iwildcam.py --mode="test" --save_folder=${SAVED_FOLDER} --input_folder=${TEST_FOLDER} --curriculum

# limits=( 100 90 70 50 20 10 0 )
limits=( 0 )

for i in "${limits[@]}"
do
	# echo "$i"
    python src/main.py --train-dataset=IWildCamIDVal --epochs=2 --lr=1e-5 --wd=0.2 --batch-size=256 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location="./datasets/data/" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_test="${SAVED_FOLDER}train.csv" --csv-img-key filepath --csv-caption-key title --exp_name="flyp_loss_prog_train_${i}_2" --progress_train --guidance=$i --scheduler=default --workers=4 --datalimit=70000 --debug --test
done
