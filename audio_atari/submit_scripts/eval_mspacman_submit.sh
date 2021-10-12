#!/bin/bash
  

screen -dmS eval_mspacman_0 bash
screen -S eval_mspacman_0 -X stuff "cd
"
screen -S eval_mspacman_0 -X stuff "conda deactivate
"
screen -S eval_mspacman_0 -X stuff "conda deactivate
"
screen -S eval_mspacman_0 -X stuff ". ./setup_trex.sh
"
screen -S eval_mspacman_0 -X stuff "sbatch slurm_scripts/eval/eval_mspacman_0.sh
"

screen -dmS eval_mspacman_1 bash
screen -S eval_mspacman_1 -X stuff "cd
"
screen -S eval_mspacman_1 -X stuff "conda deactivate
"
screen -S eval_mspacman_1 -X stuff "conda deactivate
"
screen -S eval_mspacman_1 -X stuff ". ./setup_trex.sh
"
screen -S eval_mspacman_1 -X stuff "sbatch slurm_scripts/eval/eval_mspacman_1.sh
"

screen -dmS eval_mspacman_2 bash
screen -S eval_mspacman_2 -X stuff "cd
"
screen -S eval_mspacman_2 -X stuff "conda deactivate
"
screen -S eval_mspacman_2 -X stuff "conda deactivate
"
screen -S eval_mspacman_2 -X stuff ". ./setup_trex.sh
"
screen -S eval_mspacman_2 -X stuff "sbatch slurm_scripts/eval/eval_mspacman_2.sh
"

