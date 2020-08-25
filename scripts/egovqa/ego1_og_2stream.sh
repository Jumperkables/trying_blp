#!/bin/bash

source /home/jumperkables/kable_management/python_venvs/EgoVQA/bin/activate
python /home/jumperkables/kable_management/blp_paper/egovqa/train.py \
        --split=1 \
        --memory_type=_mrm2s \
        --jobname=ego1_og_2stream
#--split=1|2|3 --memory_type=_mrm2s | _stvqa | _enc_dec | _co_mem
#python test.py --memory_type=_mrm2s | _stvqa | _enc_dec | _co_mem
