#!/bin/bash

#SBATCH --job-name=clip_ori_lyj
#SBATCH --output=clip_ori_lyj.out.%j
#SBATCH --error=clip_ori_lyj.out.%j
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

ln -s PATH_TO_YOUR_iWILDCam_DATASET ./datasets/data/iwildcam_v2.0

# python datacreation_scripts/iwildcam_ori.py 

# train
python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=256 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints_base/ --data-location="./datasets/data/" --ft_data="./datasets/csv/iwildcam_v2.0/train.csv" --csv-img-key filepath --csv-caption-key title --exp_name=iwildcam/flyp_loss_eval