#!/bin/bash
  

screen -dmS PPO_mspacman_0 bash
screen -S PPO_mspacman_0 -X stuff "cd
"
screen -S PPO_mspacman_0 -X stuff "conda deactivate
"
screen -S PPO_mspacman_0 -X stuff "conda deactivate
"
screen -S PPO_mspacman_0 -X stuff ". ./setup_trex.sh
"
screen -S PPO_mspacman_0 -X stuff "sbatch slurm_scripts/PPO/PPO_mspacman_0.sh
"

screen -dmS PPO_mspacman_1 bash
screen -S PPO_mspacman_1 -X stuff "cd
"
screen -S PPO_mspacman_1 -X stuff "conda deactivate
"
screen -S PPO_mspacman_1 -X stuff "conda deactivate
"
screen -S PPO_mspacman_1 -X stuff ". ./setup_trex.sh
"
screen -S PPO_mspacman_1 -X stuff "sbatch slurm_scripts/PPO/PPO_mspacman_1.sh
"

screen -dmS PPO_mspacman_2 bash
screen -S PPO_mspacman_2 -X stuff "cd
"
screen -S PPO_mspacman_2 -X stuff "conda deactivate
"
screen -S PPO_mspacman_2 -X stuff "conda deactivate
"
screen -S PPO_mspacman_2 -X stuff ". ./setup_trex.sh
"
screen -S PPO_mspacman_2 -X stuff "sbatch slurm_scripts/PPO/PPO_mspacman_2.sh
"
