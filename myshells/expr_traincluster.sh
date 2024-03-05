#!/bin/bash

#SBATCH --job-name=flyp_loss_train_cluster
#SBATCH --output=flyp_loss_train_cluster.out.%j
#SBATCH --error=flyp_loss_train_cluster.out.%j
#SBATCH --time=1:00:00
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

# limits=( 100 90 70 50 20 10 0 )
limits=( 0 )

for i in "${limits[@]}"
do
	# echo "$i"
    python src/main.py --train-dataset=IWildCamIDVal --epochs=4 --lr=1e-5 --wd=0.2 --batch-size=300 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location="./datasets/data/" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_test="${SAVED_FOLDER}train.csv" --csv-img-key filepath --csv-caption-key title --exp_name="flyp_loss_train_cluster_${i}" --progress_train --guidance=$i --scheduler=default --workers=4 --datalimit=70000 --cluster="loss_changes" --slurm_job_id=$SLURM_JOB_ID
done
