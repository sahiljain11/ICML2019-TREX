#!/bin/bash

#SBATCH --job-name eval_mspacman_0
#SBATCH --output=logs/eval_mspacman_0_%j.out
#SBATCH --error=logs/eval_mspacman_0_%j.err
#SBATCH --mail-user=akankshasaran@utexas.edu
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --partition dgx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time 72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
python evaluateLearnedPolicy.py --env_name mspacman --checkpointpath ppo_models/mspacman_0/checkpoints/43000
