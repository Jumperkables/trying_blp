#!/bin/bash

source /home/jumperkables/kable_management/python_venvs/EgoVQA/bin/activate
python /home/jumperkables/kable_management/blp_paper/tgif-qa/main.py --task=Action \
    --jobname=action_og_2stream \
    --num_epochs 3000 \
    --pool_type default \
    --pool_in_dims 300 300 \
    --pool_out_dim 750 \
    --pool_hidden_dim 1600
