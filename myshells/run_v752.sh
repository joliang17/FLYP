#!/bin/bash

#SBATCH --job-name=v752_3
#SBATCH --output=v752_3.out.%j
#SBATCH --error=v752_3.out.%j
#SBATCH --time=48:00:00
#SBATCH --account=cml-zhou
#SBATCH --partition=cml-zhou
#SBATCH --nodelist=cml31
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
SAVED_FOLDER="../data/metadata/clip_progress_difficult_2022_2_onlyguid_expr/"

# python datacreation_scripts/iwildcam.py --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER} --curriculum --gene_constr='../data/metadata/used_imgid/used_imgid_v2.pkl'

# while train with guid != 100, merge it with all other guid = 100 data
python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=300 --model=ViT-B/16 --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ --data-location="../data/iwildcam/" --ft_data="${SAVED_FOLDER}train.csv" --ft_data_test="${SAVED_FOLDER}curriculum.csv" --csv-img-key filepath --csv-caption-key title --exp_name="flyp_loss_v752_3" --curriculum --curriculum_epoch=15 --progress_guid --uniform_set --merge_ori --progress_metric=Prob --scheduler=default --workers=4 --slurm_job_id=$SLURM_JOB_ID

