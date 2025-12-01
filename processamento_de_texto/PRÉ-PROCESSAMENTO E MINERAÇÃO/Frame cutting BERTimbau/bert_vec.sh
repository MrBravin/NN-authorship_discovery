#!/bin/bash
#SBATCH -n 16
#SBATCH --ntasks-per-node=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
###SBATCH --mem=80000mb
#SBATCH --output=log/%x.%j.out
#SBATCH --error=log/%x.%j.err

source ~/.bashrc
conda activate tcc_data

#cd $SLURM_SUBMIT_DIR

python -u frame-cutting.py
