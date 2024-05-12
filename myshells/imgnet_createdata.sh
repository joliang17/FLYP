#!/bin/bash

#SBATCH --job-name=imgnet_guid3
#SBATCH --output=imgnet_guid3.out.%j
#SBATCH --error=imgnet_guid3.out.%j
#SBATCH --time=48:00:00
#SBATCH --account=scavenger 
#SBATCH --partition=scavenger
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G


# checking gpu status
nvidia-smi

# cd ../..
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate diffu


ORI_DATA="/fs/nexus-datasets/ImageNet"
TRAIN_FOLDER="/fs/nexus-projects/wilddiffusion/gene_diffcls/data/imagenet/train_v2"
SAVED_FOLDER="../data/imagenet/metadata/imgnet_aug_v4/"

# # train
python datacreation_scripts/imagenet_LT.py --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER} --curriculum

