#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=gpu11
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=gpu

module load conda-python/3.7
source activate gpupy36
python my_template_matcher_muzzel.py > out.txt
