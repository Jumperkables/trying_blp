#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH -x gpu[0-3]
#SBATCH --qos short
#SBATCH --job-name=v_bert
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH -o /home/crhf63/kable_management/mk8+-tvqa/dataset_paper/ncc/results/tvqa_abc_v_bert.out
#jumperkables
#crhf63
source /home/crhf63/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/crhf63/kable_management/mk8+-tvqa/main.py \
    --input_streams vcpt \
    --jobname=tvqa_abc_v_bert \
    --results_dir_base=/home/crhf63/kable_management/mk8+-tvqa/dataset_paper/ncc/results/tvqa_abc_v_bert \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 32 \
    --bert default \
    --log_freq 800 \
    --test_bsz 32 \
    --lanecheck_path /home/crhf63/kable_management/mk8+-tvqa/dataset_paper/ncc/results/tvqa_abc_v_bert/lanecheck_dict.pickle

    #--poolnonlin lrelu \
    #--pool_dropout 0.5 \
#############
#####
# REMOVE TESTRUN
####
##############