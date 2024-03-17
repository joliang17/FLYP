#!/bin/bash

#SBATCH --job-name=v0_rerun
#SBATCH --output=v0_rerun.out.%j
#SBATCH --error=v0_rerun.out.%j
#SBATCH --time=48:00:00
#SBATCH --account=scavenger 
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G



# checking gpu status
nvidia-smi

# cd ../..
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate diffu

# ln -s PATH_TO_YOUR_iWILDCam_DATASET ../data/metadata/

TRAIN_FOLDER="../data/train_new"
TEST_FOLDER="../data/test"
SAVED_FOLDER="../data/iwildcam/iwildcam_v2.0/"

# python datacreation_scripts/iwildcam.py --mode="train" --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER} --curriculum
# python datacreation_scripts/iwildcam.py --mode="test" --save_folder=${SAVED_FOLDER} --input_folder=${TEST_FOLDER} --curriculum


python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=200 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template --save=./checkpoints/ --data-location="../data/iwildcam/" --ft_data="$SAVED_FOLDER/train.csv" --csv-img-key filepath --csv-caption-key title --exp_name=v0_rerun --workers=4 --baseline