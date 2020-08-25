#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH -x gpu[0-3]
#SBATCH --qos short
#SBATCH --job-name=mcb_bert
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH -o /home/crhf63/kable_management/blp_paper/.results/tvqa/mcb_bert.out
#jumperkables
#crhf63
source /home/crhf63/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/crhf63/kable_management/blp_paper/tvqa/main.py \
    --bert default \
    --input_streams sub vcpt imagenet \
    --jobname=tvqa_svi_mcb_bert \
    --results_dir_base=/home/crhf63/kable_management/blp_paper/.results/tvqa/mcb_bert \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 16 \
    --log_freq 1600 \
    --test_bsz 100 \
    --lanecheck_path /home/crhf63/kable_management/blp_paper/.results/tvqa/mcb_bert/lanecheck_dict.pickle \
    --pool_type MCB \
    --pool_in_dims 300 300 \
    --pool_out_dim 750 \
    --pool_hidden_dim 1600
    #--poolnonlin lrelu \
    #--pool_dropout 0.5 \