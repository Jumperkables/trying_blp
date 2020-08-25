#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=mfh_svi
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/blp_paper/.results/tvqa/mfh_svi.out
#jumperkables
#crhf63
source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/jumperkables/kable_management/blp_paper/tvqa/main.py \
    --input_streams sub vcpt imagenet \
    --jobname=mfh_svi \
    --results_dir_base=/home/jumperkables/kable_management/blp_paper/.results/tvqa/mfh_svi \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 8 \
    --log_freq 3200 \
    --test_bsz 8 \
    --lanecheck_path /home/jumperkables/kable_management/blp_paper/.results/tvqa/mfh_svi/lanecheck_dict.pickle \
    --pool_type MFH \
    --pool_in_dims 300 300 \
    --pool_out_dim 750 \
    --pool_hidden_dim 1600 \
    --testrun True
    #--poolnonlin lrelu \
    #--pool_dropout 0.5 \