#!/bin/bash

#SBATCH --job-name reward_spaceinvaders 
#SBATCH --output=logs/reward_mspacman_%j.out
#SBATCH --error=logs/reward_mspacman_%j.err
#SBATCH --mail-user=asaran@cs.utexas.edu
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --partition titans
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time 72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4

python LearnAtariRewardAGC.py --env_name spaceinvaders --data_dir ./frames --reward_model_path ./learned_reward_models/spaceinvaders.params
