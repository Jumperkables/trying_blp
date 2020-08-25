#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=tucker_svi
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/blp_paper/.results/tvqa/tucker_svi.out
#jumperkables
#crhf63
source /home/crhf63/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/crhf63/kable_management/blp_paper/tvqa/main.py \
    --input_streams sub vcpt imagenet \
    --jobname=tucker_svi \
    --results_dir_base=/home/crhf63/kable_management/blp_paper/.results/tvqa/tucker_svi \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 16 \
    --log_freq 800 \
    --test_bsz 16 \
    --lanecheck_path /home/crhf63/kable_management/blp_paper/.results/tvqa/tucker_svi/lanecheck_dict.pickle \
    --pool_type Tucker \
    --pool_in_dims 300 300 \
    --pool_out_dim 750 \
    --pool_hidden_dim 1600