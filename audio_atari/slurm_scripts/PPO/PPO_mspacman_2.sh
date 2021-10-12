#!/bin/bash

#SBATCH --job-name PPO_mspacman_2 
#SBATCH --output=logs/PPO_mspacman_2_%j.out
#SBATCH --error=logs/PPO_mspacman_2_%j.err
#SBATCH --mail-user=asaran@cs.utexas.edu
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --partition titans
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time 72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4

OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=ppo_models/mspacman_2 python -m baselines.run --alg=ppo2 --env=MsPacmanNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_reward_models/mspacman.params --seed 2 --num_timesteps=5e7 --save_interval=500 --num_env 9
