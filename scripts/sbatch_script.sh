#!/bin/bash
#
#SBATCH --job-name=makeACData
#SBATCH --partition=soundbendor
#SBATCH --account=soundbendor
#SBATCH -o gen.out
#SBATCH -e gen.err

module load slurm
source ../env/bin/activate
python3 gen_ac_dataset.py

#SBATCH --mem=15G

