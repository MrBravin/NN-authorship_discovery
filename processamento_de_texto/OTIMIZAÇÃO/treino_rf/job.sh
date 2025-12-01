#!/bin/bash
#SBATCH -n 32
#SBATCH --ntasks-per-node=32
#SBATCH -p batch-AMD
###SBATCH --mem=80000mb
#SBATCH --output=log/%x.%j.out
#SBATCH --error=log/%x.%j.err

source ~/.bashrc
conda activate tcc_data

#cd $SLURM_SUBMIT_DIR

python -u treino_rf_.py
