#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=tvqa_svi_noqs
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/blp_paper/.results/tvqa/tvqa_svi_noqs.out
#jumperkables
#crhf63
source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/jumperkables/kable_management/blp_paper/tvqa/main.py \
    --input_streams sub vcpt imagenet \
    --jobname=tvqa_svi_noqs \
    --results_dir_base=/home/jumperkables/kable_management/blp_paper/.results/tvqa/tvqa_svi_noqs \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 32 \
    --log_freq 800 \
    --test_bsz 100 \
    --lanecheck_path /home/jumperkables/kable_management/blp_paper/.results/tvqa/tvqa_svi_noqs/lanecheck_dict.pickle \
    --pool_type default \
    --noqs
    #--poolnonlin lrelu \
    #--pool_dropout 0.5 \