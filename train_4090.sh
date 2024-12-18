#!/bin/bash
#SBATCH --job-name=1
#SBATCH --partition=4090
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --output=debug_4090.out   # 标准输出文件名

source /share/anaconda3/etc/profile.d/conda.sh
conda  activate greenroof
srun python train.py --batch-size 70
