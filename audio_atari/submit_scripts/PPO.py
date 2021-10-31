import pathlib

TEMPLATE = f"""#!/bin/bash
cd
. ./setup_trex.sh

OPENAI_LOG_FORMAT=\'stdout,log,csv,tensorboard\' OPENAI_LOGDIR=ppo_models/"""


# Defines a grid search over these parameters.
PARAMS = {
    'seed': [0,1,2],
    'exp': ['rl_cgl_cal_0.5-0.1-0.4'],
    # 'exp': ['cal', 'cgl', 'rl_cal', 'rl_cgl', 'cgl_cal', 'rl_cgl_cal'],
    'env': ['mspacman','spaceinvaders','seaquest'],
}


def get_job(seed, exp, env):
    """
    build your shell script here.
    """

    if env=='mspacman':
        env_id = 'MsPacman'
    if env=='seaquest':
        env_id = 'Seaquest'
    if env=='spaceinvaders':
        env_id = 'SpaceInvaders'

    extra = '_'+exp

    if exp=='':
        reward_dir = 'learned_reward_models/'
    elif exp=='cal':
        reward_dir = 'multimodal_reward_models/'+env+'_CAL_only/'
    elif exp=='cgl':
        reward_dir = 'multimodal_reward_models/'+env+'_CGL_only/'
    else:
        reward_dir = './multimodal_reward_models/'+env+extra+'/'


    shell_script_str = TEMPLATE
    shell_script_str += f'{env}'
    shell_script_str += extra+'_'
    shell_script_str += f'{seed}'
    shell_script_str += ' python -m baselines.run --alg=ppo2 --env='
    shell_script_str += env_id
    shell_script_str += 'NoFrameskip-v4 --custom_reward pytorch --custom_reward_path '+reward_dir
    shell_script_str += f'{env}'
    shell_script_str += '.params --seed '
    shell_script_str += f'{seed}'
    shell_script_str += ' --num_timesteps=5e7 --save_interval=500 --num_env 9' 
    

    return shell_script_str