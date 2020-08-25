#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-6]
#SBATCH --qos long-high-prio
#SBATCH --job-name=c3d_extract
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH -o /home/crhf63/c3d_extract.out
source /home/crhf63/kable_management/python_venvs/EgoVQA/bin/activate
python feature_extractor_frm.py --gpu 
