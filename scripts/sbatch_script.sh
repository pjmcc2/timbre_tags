#!/bin/bash
#
#SBATCH --partition=soundbendor
#SBATCH --account=soundbendor
#SBATCH --job-name=tt_no_audio
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=6G
#SBATCH -o scripts/gen.out
#SBATCH -e scripts/gen.err

module load slurm
source env/bin/activate
python train_no_audio.py

# arguments for train_text_classifier: C,L,M

#SBATCH --partition=soundbendor
#SBATCH --account=soundbendor

#SBATCH --time limit simethign
#SBATCH --partition=dgx2