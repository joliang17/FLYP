#!/bin/bash

TRAIN_FOLDER="../data/train_new"
TEST_FOLDER="../data/test"
SAVED_FOLDER="../data/metadata/clip_newcurri/"

python datacreation_scripts/iwildcam.py --mode="curriculum" --save_folder=${SAVED_FOLDER} --input_folder=${TRAIN_FOLDER} --curriculum --total_train