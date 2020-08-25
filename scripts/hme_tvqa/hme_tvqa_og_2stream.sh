#!/bin/bash

source /home/jumperkables/kable_management/python_venvs/EgoVQA/bin/activate
python /home/jumperkables/kable_management/blp_paper/hme_tvqa/main.py --task=Trans \
    --input_streams sub vcpt imagenet \
    --bsz 64 \
    --test_bsz 64 \
    --log_freq 400 \
    --jobname=hme_tvqa_og_2stream \
    --num_epochs 3000 \
    --pool_type default \
    --pool_in_dims 300 300 \
    --pool_out_dim 750 \
    --pool_hidden_dim 1600 \
    --no_core_driver \
    --num_workers 1 \
    --hard_restrict
