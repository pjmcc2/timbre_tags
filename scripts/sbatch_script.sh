#!/bin/bash
#
#SBATCH --job-name=CLAP_text
#SBATCH --gres=gpu:1
#SBATCH --mem=6G
#SBATCH -o gen.out
#SBATCH -e gen.err

module load slurm
source env/bin/activate
python train_text_classifier.py M

# arguments for train_text_classifier: C,L,M

#SBATCH --partition=soundbendor
#SBATCH --account=soundbendor