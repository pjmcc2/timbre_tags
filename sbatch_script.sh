#!/bin/bash
#
#SBATCH --partition=soundbendor
#SBATCH --account=soundbendor
#SBATCH --job-name=embed_wav
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=15G
#SBATCH --time=23:59:59
#SBATCH -o sbres.out
#SBATCH -e sbres.err

module load slurm
source env/bin/activate
python -m data.utils.embed_wavcaps --dataset_name=ac

# arguments for train_text_classifier: C,L,M

#SBATCH --partition=soundbendor
#SBATCH --account=soundbendor

#SBATCH --time limit simethign
#SBATCH --partition=dgx2