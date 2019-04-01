#!/bin/sh
#SBATCH --mem=40G
#SBATCH -o train.out
#SBATCH -e error.out
#SBATCH -p dsplus-gpu
#SBATCH --account=plusds
#SBATCH --gres=gpu:1

source activate base 
python inception_v3_multiInput_SGD.py
