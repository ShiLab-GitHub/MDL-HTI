#!/bin/bash
#SBATCH -J mv-hit
#SBATCH --cpus-per-task=12
##SBATCH --gres=gpu:2
#SBATCH -p gpu
#SBATCH -t 5-00:00:00
          
module add cuda/10.1 

python main.py --dataset ddb --clustering 1 --hidden_dim 128 --lr 0.0001 --weight_decay 0.0005 --num_heads 8 --num_layers 3 --dropout 0.1 --context_hops 12 --max_path_len 10 --path_samples 12 --cluster_coeff 0.25 --num_clusters 2 --gpu 2
