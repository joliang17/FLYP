#!/bin/bash

#SBATCH --job-name=clip_base_lyj
#SBATCH --output=clip_base_lyj.out.%j
#SBATCH --error=clip_base_lyj.out.%j
#SBATCH --time=12:00:00
#SBATCH --account=scavenger 
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G


# checking gpu status
nvidia-smi

# cd ../..
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate diffu

# ln -s PATH_TO_YOUR_iWILDCam_DATASET ../data/metadata/

TRAIN_FOLDER="../data/train_new"
TEST_FOLDER="../data/test"
SAVED_FOLDER="../data/metadata/clip_base/"

python datacreation_scripts/iwildcam.py --mode="train" --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER}
python datacreation_scripts/iwildcam.py --mode="test" --save_folder=${SAVED_FOLDER} --input_folder=${TEST_FOLDER}

# python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=256 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location="../data/iwildcam" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_test="${SAVED_FOLDER}test.csv" --csv-img-key filepath --csv-caption-key title --exp_name=iwildcam/flyp_loss --curriculum

python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=128 --model=ViT-B/16 --eval-datasets=IWildCamIDVal --template=iwildcam_template  --save=./checkpoints_base/ --data-location="../data/iwildcam" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_test="${SAVED_FOLDER}test.csv" --csv-img-key filepath --csv-caption-key title --exp_name=iwildcam/flyp_loss_base --self_data