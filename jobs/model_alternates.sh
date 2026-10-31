#!/bin/bash

# Alternative models for the comparison tables. Submit from the repository root: qsub jobs/model_alternates.sh
#$ -cwd
#$ -pe smp 8
#$ -q gpu@@crc_gpu
#$ -l gpu_card=1
#$ -N train_ens_world_1
#$ -t 1-10

export OMP_NUM_THREADS=1
export TF_GPU_ALLOCATOR=cuda_malloc_async

module load cuda
module load python

# 10 tasks = 5 models x 2 outcomes
#   task 1,2  -> rf        (odd = Conflict, even = Unrest)
#   task 3,4  -> lgbm
#   task 5,6  -> catboost
#   task 7,8  -> et
#   task 9,10 -> logreg
MODELS=(rf lgbm catboost et logreg)
MODEL=${MODELS[$(( (SGE_TASK_ID - 1) / 2 ))]}

echo "Task ${SGE_TASK_ID}: model=${MODEL}"
python3 scripts/model_alternates.py --model ${MODEL}
