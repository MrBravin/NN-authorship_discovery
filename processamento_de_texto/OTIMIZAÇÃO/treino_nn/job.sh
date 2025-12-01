#!/bin/bash
#SBATCH -n 8
#SBATCH --ntasks-per-node=8
#SBATCH -p gpu
#SBATCH --gres=gpu:4090:2
###SBATCH --mem=80000mb
#SBATCH --output=log/%x.%j.out
#SBATCH --error=log/%x.%j.err

source ~/.bashrc
conda activate tcc_data

#cd $SLURM_SUBMIT_DIR

python -u siamese_mlp_optuna.py
