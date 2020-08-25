#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=OG_svi
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/blp_paper/.results/tvqa/OG_svi.out
#jumperkables
#crhf63
source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/jumperkables/kable_management/blp_paper/tvqa/main.py \
    --input_streams sub vcpt imagenet \
    --jobname=OG_svi \
    --results_dir_base=/home/jumperkables/kable_management/blp_paper/.results/tvqa/tvqa_abc_svi \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 64 \
    --log_freq 400 \
    --test_bsz 64 \
    --lanecheck_path /home/jumperkables/kable_management/blp_paper/.results/tvqa/tvqa_abc_svi/lanecheck_dict.pickle \
    --pool_type default
    #--poolnonlin lrelu \
    #--pool_dropout 0.5 \