#!/bin/bash
#
#SBATCH --partition=dgx2
#SBATCH --job-name=CLAP_text
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=5G
#SBATCH -o scripts/gen.out
#SBATCH -e scripts/gen.err

module load slurm
source env/bin/activate
python train_text_classifier.py M

# arguments for train_text_classifier: C,L,M

#SBATCH --partition=soundbendor
#SBATCH --account=soundbendor