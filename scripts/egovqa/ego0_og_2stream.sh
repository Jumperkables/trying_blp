#!/bin/bash

source /home/jumperkables/kable_management/python_venvs/EgoVQA/bin/activate
python /home/jumperkables/kable_management/blp_paper/egovqa/train.py \
        --split=0 \
        --memory_type=_mrm2s \
        --jobname=ego0_og_2stream \
        --pool_type default \
        --pool_in_dims 300 300 \
        --pool_out_dim 750 \
        --pool_hidden_dim 1600 \
#--split=1|2|3 --memory_type=_mrm2s | _stvqa | _enc_dec | _co_mem
#python test.py --memory_type=_mrm2s | _stvqa | _enc_dec | _co_mem
