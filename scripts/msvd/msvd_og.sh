#!/bin/bash

source /home/jumperkables/kable_management/python_venvs/EgoVQA/bin/activate
python /home/jumperkables/kable_management/blp_paper/msvd-qa/main.py \
        --jobname=blp_msvd \
        --pool_type default
