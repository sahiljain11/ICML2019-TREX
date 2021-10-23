
# env_names = [['seaquest','Seaquest'],['spaceinvaders','SpaceInvaders'],['mspacman','MsPacman'],['enduro','Enduro'],['montezumarevenge','MontezumaRevenge']]
# env = ['seaquest','Seaquest']
# env = ['spaceinvaders','SpaceInvaders']
env = ['mspacman','MsPacman']
# env = ['enduro','Enduro']
# env = ['montezumarevenge','MontezumaRevenge']



# extra = 'cal_'

extra, extra_option = '_rl_cgl', 'rl_cgl'
# extra, extra_option = '_rl_cal', 'rl_cal'
# extra, extra_option = '_cgl_cal', 'cgl_cal'
# extra, extra_option = '_rl_cgl_cal', 'rl_cgl_cal'

# reward_dir = 'learned_reward_models/'
# reward_dir = 'multimodal_reward_models/'+env[0]+'_CAL_only/'
reward_dir = './multimodal_reward_models/'+env[0]+extra+'/'


# server = 'dgx'
server = 'titans' 
seeds = ['0','1','2']

for seed in seeds:
    bash_file_name = 'PPO/PPO'+extra+'_'+env[0]+'_'+seed+'.sh'
    f = open(bash_file_name,'w')
    f.write("#!/bin/bash\n\n")

    f.write('#SBATCH --job-name PPO'+extra+'_'+env[0]+'_'+seed+'\n')
    f.write('#SBATCH --output=logs/PPO'+extra+'_'+env[0]+'_'+seed+'_%j.out\n')
    f.write('#SBATCH --error=logs/PPO'+extra+'_'+env[0]+'_'+seed+'_%j.err\n')
    f.write('#SBATCH --mail-user=akankshasaran@utexas.edu\n')
    f.write('#SBATCH --mail-type=END,FAIL,REQUEUE\n')
    f.write('#SBATCH --partition '+server+'\n')
    f.write('#SBATCH --nodes=1\n')
    f.write('#SBATCH --ntasks-per-node=1\n')
    f.write('#SBATCH --time 72:00:00\n')
    f.write('#SBATCH --gres=gpu:1\n')
    f.write('#SBATCH --mem=30G\n')
    f.write('#SBATCH --cpus-per-task=4\n')

    # f.write('OPENAI_LOG_FORMAT=\'stdout,log,csv,tensorboard\' OPENAI_LOGDIR=ppo_models/'+env[0]+'_'+extra+seed+' python -m baselines.run --alg=ppo2 --env='+env[1]+'NoFrameskip-v4 --custom_reward pytorch --custom_reward_path '+reward_dir+env[0]+'.params --seed '+seed+' --num_timesteps=5e7 --save_interval=500 --num_env 9')
    f.write('OPENAI_LOG_FORMAT=\'stdout,log,csv,tensorboard\' OPENAI_LOGDIR=ppo_models/'+env[0]+extra+'_'+seed+' python -m baselines.run --alg=ppo2 --env='+env[1]+'NoFrameskip-v4 --custom_reward pytorch --custom_reward_path '+reward_dir+env[0]+'.params --seed '+seed+' --num_timesteps=5e7 --save_interval=500 --num_env 9')

    f.close()