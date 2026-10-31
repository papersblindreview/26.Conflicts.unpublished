#!/bin/bash
# Main XGBoost model. Task 1 = Conflict, task 2 = Unrest. Submit from the repository root: qsub jobs/train_xgb.sh
# The job name encodes <continent>_<config line> and is read by the script (here: world, line 1 of model_configs.txt).
# Note the printed best_iteration + 1 for each outcome; bootstrap_shap.sh needs them.

#$ -cwd
#$ -pe smp 8
#$ -q gpu@@crc_gpu
#$ -l gpu_card=1
#$ -N train_xgb_world_1
#$ -t 1-2

export OMP_NUM_THREADS=1
module load cuda
module load python

python3 scripts/model_xgb.py
