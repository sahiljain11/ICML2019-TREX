#!/bin/bash


screen -dmS reward_mspacman bash
screen -S reward_mspacman -X stuff "cd
"
screen -S reward_mspacman -X stuff "conda deactivate
"
screen -S reward_mspacman -X stuff "conda deactivate
"
screen -S reward_mspacman -X stuff ". ./setup_trex.sh
"
screen -S reward_mspacman -X stuff "sbatch slurm_scripts/reward_mspacman.sh
"

screen -dmS reward_spaceinvaders bash
screen -S reward_spaceinvaders -X stuff "cd
"
screen -S reward_spaceinvaders -X stuff "conda deactivate
"
screen -S reward_spaceinvaders -X stuff "conda deactivate
"
screen -S reward_spaceinvaders -X stuff ". ./setup_trex.sh
"
screen -S reward_spaceinvaders -X stuff "sbatch slurm_scripts/reward_spaceinvaders.sh
"
