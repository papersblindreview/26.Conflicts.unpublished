#!/bin/bash
# Build the 50 date partitions (dataset + climate + vdem stages). Submit from the repository root: qsub jobs/preprocessing.sh
# Run the one-off download first:  python scripts/preprocessing.py --stage download --var all

#$ -cwd
#$ -pe smp 8
#$ -N preprocess_world_1
#$ -t 1-50

module load python

python3 scripts/preprocessing.py
