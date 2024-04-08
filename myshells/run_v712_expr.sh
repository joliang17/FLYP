#!/bin/bash

#SBATCH --job-name=v712_expr
#SBATCH --output=v712_expr.out.%j
#SBATCH --error=v712_expr.out.%j
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


TRAIN_FOLDER="../data/train_new"
TEST_FOLDER="../data/test"
SAVED_FOLDER="../data/metadata/clip_progress_difficult_2022_randseed_expr/"

# python datacreation_scripts/iwildcam.py --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER} --curriculum --gene_constr='../data/metadata/used_imgid/used_imgid_v2.pkl'

python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=200 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location="../data/iwildcam/" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_test="${SAVED_FOLDER}curriculum.csv" --csv-img-key filepath --csv-caption-key title --exp_name="progress_expr" --curriculum --curriculum_epoch=15 --progress_guid --uniform_set --progress_metric=Prob --proportion --scheduler=default --workers=4 --slurm_job_id=$SLURM_JOB_ID --debug

