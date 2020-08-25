#!/bin/bash

source /home/jumperkables/kable_management/python_venvs/EgoVQA/bin/activate
python /home/jumperkables/kable_management/blp_paper/msvd-qa/main.py \
    --jobname=blp_msvd_mcb \
    --pool_type MCB \
    --pool_in_dims 512 512 \
    --pool_out_dim 1024 \
    --pool_hidden_dim 1600 
