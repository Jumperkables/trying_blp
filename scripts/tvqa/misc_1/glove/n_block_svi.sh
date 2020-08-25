#!/bin/bash
#SBATCH -N 1
#SBATCH -p res-gpu-small
#SBATCH -c 4
#SBATCH -t 7-00:00
##SBATCH -x gpu[0-6]
#SBATCH --qos long-high-prio
#SBATCH --job-name=block_glove
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH -o /home/crhf63/kable_management/blp_paper/.results/tvqa/block_glove.out
#jumperkables
#crhf63
source /home/crhf63/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/crhf63/kable_management/blp_paper/tvqa/main.py \
    --input_streams sub vcpt imagenet \
    --jobname=tvqa_svi_block_glove \
    --results_dir_base=/home/crhf63/kable_management/blp_paper/.results/tvqa/block_glove \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 4 \
    --log_freq 6400 \
    --test_bsz 50 \
    --lanecheck_path /home/crhf63/kable_management/blp_paper/.results/tvqa/block_glove/lanecheck_dict.pickle \
    --pool_type Block \
    --pool_in_dims 300 300 \
    --pool_out_dim 750 \
    --pool_hidden_dim 1600
    #--poolnonlin lrelu \
    #--pool_dropout 0.5 \