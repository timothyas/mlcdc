#!/bin/bash

#SBATCH -J reorg
#SBATCH -o slurm_reorg.%j.out
#SBATCH -e slurm_reorg.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --partition=orion
#SBATCH -t 04:00:00

eval "$(conda shell.bash hook)"
conda activate mlcdc
python reorg.py
