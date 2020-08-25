#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=tvqa-svi_bert
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/blp_paper/.results/tvqa/tvqa-svi_bert.out
#jumperkables
#crhf63
source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/jumperkables/kable_management/blp_paper/tvqa/main.py \
    --input_streams sub vcpt imagenet \
    --jobname=tvqa_abc_svi_bert \
    --results_dir_base=/home/jumperkables/kable_management/blp_paper/.results/tvqa/tvqa_abc_svi_bert \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bert default \
    --bsz 32 \
    --log_freq 800 \
    --test_bsz 32 \
    --lanecheck_path /home/jumperkables/kable_management/blp_paper/.results/tvqa/tvqa_abc_svi_bert/lanecheck_dict.pickle \
    --pool_type default
    #--poolnonlin lrelu \
    #--pool_dropout 0.5 \