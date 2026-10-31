#!/bin/bash
# Group-bootstrap replicates (GPU). Submit from the repository root: qsub jobs/bootstrap_shap.sh
# Each task runs --n_rep replicates for one outcome (odd tasks = Conflict, even = Unrest), so
# -t 1-8 with --n_rep 10 gives 4 chunks x 10 = 40 replicates per outcome. Finished replicates are skipped on resubmission.
# --rounds_*: best_iteration + 1 printed by the main run (jobs/train_xgb.sh).
# --n_expl: size of the fixed explanation sample; results/plot_data_*.csv holds at most 300,000 rows.

#$ -cwd
#$ -pe smp 8
#$ -q gpu@@crc_gpu
#$ -l gpu_card=1
#$ -N boot_shap_world_1
#$ -t 1-8

export OMP_NUM_THREADS=1
module load cuda
module load python

python3 scripts/bootstrap_shap.py --n_rep 10 --n_expl 300000 --rounds_conflict 1120 --rounds_unrest 505
