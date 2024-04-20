#!/bin/bash

#SBATCH --job-name=test_diff
#SBATCH --partition=dgx_normal_q
#SBATCH --time=20:00:00
#SBATCH -A HPCBIGDATA2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48

module reset
module load Anaconda3/2020.11

source activate myenvs/diff_env

python main.py