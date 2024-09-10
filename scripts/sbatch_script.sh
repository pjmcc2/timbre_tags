#!/bin/bash
#
#SBATCH --job-name=makeACData
#SBATCH --partition=soundbendor
#SBATCH --account=soundbendor
#SBATCH -o gen.out
#SBATCH -e gen.err

module load slurm
source ../env/bin/activate
python3 train_text_classifier C

# arguments for train_text_classifier: C,L,M, or P

