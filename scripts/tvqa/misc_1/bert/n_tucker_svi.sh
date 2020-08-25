#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH -t 7-00:00
#SBATCH -x gpu[0-6]
#SBATCH --qos long-high-prio
#SBATCH --job-name=tucker_bert
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH -o /home/crhf63/kable_management/blp_paper/.results/tvqa/tucker_bert.out
#jumperkables
#crhf63
source /home/crhf63/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/crhf63/kable_management/blp_paper/tvqa/main.py \
    --bert default \
    --input_streams sub vcpt imagenet \
    --jobname=tvqa_svi_tucker_bert \
    --results_dir_base=/home/crhf63/kable_management/blp_paper/.results/tvqa/tucker_bert \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 2 \
    --log_freq 12800 \
    --test_bsz 5 \
    --lanecheck_path /home/crhf63/kable_management/blp_paper/.results/tvqa/tucker_bert/lanecheck_dict.pickle \
    --pool_type Tucker \
    --pool_in_dims 300 300 \
    --pool_out_dim 750 \
    --pool_hidden_dim 1600
    #--poolnonlin lrelu \
    #--pool_dropout 0.5 \ 4/6400