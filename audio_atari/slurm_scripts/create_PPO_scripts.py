import argparse

def write_script(args):
    if args.env=='seaquest':
        env = ['seaquest','Seaquest'] 
    elif args.env=='spaceinvaders':
        env = ['spaceinvaders','SpaceInvaders']
    elif args.env=='mspacman':
        env = ['mspacman','MsPacman']
    elif args.env=='enduro':
        env = ['enduro','Enduro']
    elif args.env=='revenge':
        env = ['montezumarevenge','MontezumaRevenge']

    extra = '_'+args.exp

    if args.exp=='':
        reward_dir = 'learned_reward_models/'
    elif args.exp=='cal':
        reward_dir = 'multimodal_reward_models/'+env[0]+'_CAL_only/'
    elif args.exp=='cgl':
        reward_dir = 'multimodal_reward_models/'+env[0]+'_CGL_only/'
    else:
        reward_dir = './multimodal_reward_models/'+env[0]+extra+'/'
    
    server = args.cluster
    seeds = ['0','1','2']

    for seed in seeds:
        bash_file_name = 'PPO/PPO'+'_'+env[0]+extra+'_'+seed+'.sh'
        f = open(bash_file_name,'w')
        f.write("#!/bin/bash\n\n")

        f.write('#SBATCH --job-name PPO'+'_'+env[0]+extra+'_'+seed+'\n')
        f.write('#SBATCH --output=logs/PPO'+'_'+env[0]+extra+'_'+seed+'_%j.out\n')
        f.write('#SBATCH --error=logs/PPO'+'_'+env[0]+extra+'_'+seed+'_%j.err\n')
        f.write('#SBATCH --mail-user=akanksha.saran.iitj@gmail.com\n')
        f.write('#SBATCH --mail-type=END,FAIL,REQUEUE\n')
        f.write('#SBATCH --partition '+server+'\n')
        f.write('#SBATCH --nodes=1\n')
        f.write('#SBATCH --ntasks-per-node=2\n')
        f.write('#SBATCH --time 72:00:00\n')
        f.write('#SBATCH --gres=gpu:1\n')
        f.write('#SBATCH --mem=40G\n')
        f.write('#SBATCH --cpus-per-task=4\n')

        # f.write('OPENAI_LOG_FORMAT=\'stdout,log,csv,tensorboard\' OPENAI_LOGDIR=ppo_models/'+env[0]+'_'+extra+seed+' python -m baselines.run --alg=ppo2 --env='+env[1]+'NoFrameskip-v4 --custom_reward pytorch --custom_reward_path '+reward_dir+env[0]+'.params --seed '+seed+' --num_timesteps=5e7 --save_interval=500 --num_env 9')
        f.write('OPENAI_LOG_FORMAT=\'stdout,log,csv,tensorboard\' OPENAI_LOGDIR=ppo_models/'+env[0]+extra+'_'+seed+' python -m baselines.run --alg=ppo2 --env='+env[1]+'NoFrameskip-v4 --custom_reward pytorch --custom_reward_path '+reward_dir+env[0]+'.params --seed '+seed+' --num_timesteps=5e7 --save_interval=500 --num_env 9')

        f.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', default='mspacman', help='Select the environment name to run, i.e. mspacman')
    parser.add_argument('--exp', default='cal', help='Select the exp to run [cal, cgl, rl_cal, rl_cgl, cgl_cal, rl_cgl_cal, rl_cgl_correct,\
        rl_cal_ann, rl_cal_pase, rl_cal_prosody_ann_pitch, rl_cal_prosody_ann_energy, rl_cal_prosody_pase_pitch, rl_cal_prosody_pase_energy]')
    parser.add_argument('--cluster', default='titans', help='Select the cluster to run jobs [titans, dgx]')

    args = parser.parse_args()

    write_script(args)