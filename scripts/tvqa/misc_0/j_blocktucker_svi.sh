#!/bin/bash
#!/bin/bash
#SBATCH --partition=part0
#SBATCH --job-name=blocktucker_svi
#SBATCH --ntasks=6
#SBATCH --gres=gpu:1
#SBATCH -o /home/jumperkables/kable_management/blp_paper/.results/tvqa/blocktucker_svi.out
#jumperkables
#crhf63
source /home/jumperkables/kable_management/python_venvs/mk8-tvqa/bin/activate
python -W ignore /home/jumperkables/kable_management/blp_paper/tvqa/main.py \
    --input_streams sub vcpt imagenet \
    --jobname=blocktucker_svi \
    --results_dir_base=/home/jumperkables/kable_management/blp_paper/.results/tvqa/blocktucker_svi \
    --modelname=tvqa_abc_bert_nofc \
    --lrtype radam \
    --bsz 16 \
    --log_freq 1600 \
    --test_bsz 128 \
    --lanecheck_path /home/jumperkables/kable_management/blp_paper/.results/tvqa/blocktucker_svi/lanecheck_dict.pickle \
    --pool_type BlockTucker \
    --pool_in_dims 300 300 \
    --pool_out_dim 750 \
    --pool_hidden_dim 1600