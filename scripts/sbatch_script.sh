#!/bin/bash
#
#SBATCH --partition=soundbendor
#SBATCH --account=soundbendor
#SBATCH --job-name=no_audio_sig
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=6G
#SBATCH --time=23:59:59
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