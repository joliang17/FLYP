#!/bin/bash

#SBATCH --job-name=guid_train
#SBATCH --output=guid_train.out.%j
#SBATCH --error=guid_train.out.%j
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
SAVED_FOLDER="../data/metadata/clip_progress_difficult_2022_5_analysis/"

limits=( 100 90 70 50 20 10 0 )


for i in "${limits[@]}"
do
	# echo "$i"
    python src/main.py --train-dataset=IWildCamIDVal --epochs=3 --lr=1e-5 --wd=0.2 --batch-size=200 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template --save=./checkpoints/ --data-location="../data/iwildcam/" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_test="${SAVED_FOLDER}train.csv" --csv-img-key filepath --csv-caption-key title --exp_name="guid_expr_${i}" --scheduler=default --progress_guid --progress_metric=Prob --curriculum --curriculum_epoch=10 --slurm_job_id=$SLURM_JOB_ID --debug --datalimit=70000 --guidance=$i --workers=4 
done


