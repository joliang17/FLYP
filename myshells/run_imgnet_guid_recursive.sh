#!/bin/bash

#SBATCH --job-name=imgnet_guid
#SBATCH --output=imgnet_guid.out.%j
#SBATCH --error=imgnet_guid.out.%j
#SBATCH --time=48:00:00
#SBATCH --account=cml-zhou
#SBATCH --partition=cml-zhou
#SBATCH --qos=cml-medium
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

# checking gpu status
nvidia-smi

# cd ../..
source /fs/nexus-scratch/yliang17/miniconda3/bin/activate diffu


ORI_DATA="/fs/nexus-datasets/ImageNet"
TRAIN_FOLDER="/fs/nexus-projects/wilddiffusion/gene_diffcls/data/imagenet/train_new"
SAVED_FOLDER="../data/imagenet/metadata/imgnet_aug_v4/"

# # # train
# python datacreation_scripts/imagenet_LT.py --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER} --curriculum
# # # test
# python datacreation_scripts/imagenet_LT.py --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER} --test

# while train with guid != 100, merge it with all other guid = 100 data
# python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=300 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location="../data/iwildcam/" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_test="${SAVED_FOLDER}curriculum.csv" --csv-img-key filepath --csv-caption-key title --exp_name="flyp_loss_v752_2" --curriculum --curriculum_epoch=15 --progress_guid --uniform_set --merge_ori --progress_metric=Prob --scheduler=default --workers=4 --slurm_job_id=$SLURM_JOB_ID --wandb_tag="select_guid"

python src/main.py --train-dataset=ImageNet --epochs=20 --lr=1e-5 --wd=0.1 --batch-size=300 --model=ViT-B/16 --eval-datasets=ImageNetLT  --template=openai_imagenet_template_reduced  --save=/fs/nexus-projects/wilddiffusion/gene_diffcls/FYLP/checkpoints/ --data-location="${ORI_DATA}" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_curri="${SAVED_FOLDER}curriculum.csv"  --ft_data_test="${SAVED_FOLDER}test.csv" --csv-img-key filepath --csv-caption-key title --exp_name="flyp_loss_imgnet_recur" --curriculum --curriculum_epoch=15 --merge_ori --progress_metric=Prob --slurm_job_id=$SLURM_JOB_ID --wandb_tag="image_net"

