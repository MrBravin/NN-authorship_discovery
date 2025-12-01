#!/bin/bash
#SBATCH -n 32
#SBATCH --ntasks-per-node=32
#SBATCH -p batch-AMD
###SBATCH --mem=80000mb
#SBATCH --output=log_cpu/%x.%j.out
#SBATCH --error=log_cpu/%x.%j.err

source ~/.bashrc
conda activate tcc_data

#cd $SLURM_SUBMIT_DIR

python -u treino_optuna_svm_final2.py
