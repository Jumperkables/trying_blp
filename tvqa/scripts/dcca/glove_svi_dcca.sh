#!/bin/bash

source /home/jumperkables/kable_management/python_venvs/tvqa/bin/activate
python -W ignore /home/jumperkables/kable_management/blp_paper/tvqa/main.py \
    --input_streams sub vcpt imagenet \
    --jobname=blp_tvqa_dcca_svi_glove \
    --results_dir_base=/home/jumperkables/kable_management/blp_paper/.results/tvqa/og_dcca_glove \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 8 \
    --log_freq 3200 \
    --test_bsz 50 \
    --lanecheck_path /home/jumperkables/kable_management/blp_paper/.results/tvqa/og_dcca_glove/lanecheck_dict.pickle \
    --pool_type default \
    --deep_cca \
    --deep_cca_layers 2

