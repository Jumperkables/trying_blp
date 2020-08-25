#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH -t 2-00:00
#SBATCH -x gpu[0-6]
#SBATCH --qos short
#SBATCH --job-name=dcca_nosub
#SBATCH --mem=14G
#SBATCH --gres=gpu:1
#SBATCH -o /home/crhf63/kable_management/blp_paper/.results/tvqa/og_dcca_nosub.out
#crhf63
#crhf63
source /home/crhf63/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/crhf63/kable_management/blp_paper/tvqa/main.py \
    --input_streams vcpt imagenet \
    --jobname=blp_tvqa_dcca_vi \
    --results_dir_base=/home/crhf63/kable_management/blp_paper/.results/tvqa/og_dcca_nosub \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 16 \
    --log_freq 1600 \
    --test_bsz 100 \
    --lanecheck_path /home/crhf63/kable_management/blp_paper/.results/tvqa/og_dcca_nosub/lanecheck_dict.pickle \
    --pool_type default \
    --deep_cca \
    --deep_cca_layers 2
    #--poolnonlin lrelu \
    #--pool_dropout 0.5 \
#############
#####
# REMOVE TESTRUN
####
##############
