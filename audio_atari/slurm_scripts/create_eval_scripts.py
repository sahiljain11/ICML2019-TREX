# env_names = [['seaquest','Seaquest'],['spaceinvaders','SpaceInvaders'],['mspacman','MsPacman'],['enduro','Enduro'],['montezumarevenge','MontezumaRevenge']]
env = ['seaquest','Seaquest']
# env = ['spaceinvaders','SpaceInvaders']
# env = ['mspacman','MsPacman']
# env = ['enduro','Enduro']
# env = ['montezumarevenge','MontezumaRevenge']

server = 'titans' #'dgx'
seeds = ['0','1','2']

# extra = '_cal'
# extra = '_rl_cgl'
# extra = '_cgl'
# extra = '_rl_cal'
# extra = '_rl_cgl_cal'
# extra = '_cgl_cal'
# extra = '_rl_cgl_cal_0.5-0.1-0.4'
# extra = '_rl_cgl_cal_0.1-0.8-0.1'
# extra = '_rl_cgl_cal_0.35-0.05-0.6'
# extra = '_rl_cgl_cal_0.1-0.45-0.45'
# extra = '_rl_cgl_cal_0.4-0.2-0.4'
extra = '_rl_cgl_cal_0.9-0.05-0.05'

for seed in seeds:
    bash_file_name = 'eval/eval_'+env[0]+extra+'_'+seed+'.sh'
    f = open(bash_file_name,'w')
    f.write("#!/bin/bash\n\n")

    f.write('#SBATCH --job-name eval_'+env[0]+extra+'_'+seed+'\n')
    f.write('#SBATCH --output=logs/eval_'+env[0]+extra+'_'+seed+'_%j.out\n')
    f.write('#SBATCH --error=logs/eval_'+env[0]+extra+'_'+seed+'_%j.err\n')
    f.write('#SBATCH --mail-user=akankshasaran@utexas.edu\n')
    f.write('#SBATCH --mail-type=END,FAIL,REQUEUE\n')
    f.write('#SBATCH --partition '+server+'\n')
    f.write('#SBATCH --nodes=1\n')
    f.write('#SBATCH --ntasks-per-node=1\n')
    f.write('#SBATCH --time 72:00:00\n')
    f.write('#SBATCH --gres=gpu:1\n')
    f.write('#SBATCH --mem=10G\n')
    f.write('#SBATCH --cpus-per-task=4\n')

    # f.write('python evaluateLearnedPolicy.py --env_name '+env[0]+' --checkpointpath ppo_models/'+env[0]+'_'+seed+'/checkpoints/43000')
    f.write('python evaluateLearnedPolicy.py --env_name '+env[0]+' --checkpointpath ppo_models/'+env[0]+extra+'_'+seed+'/checkpoints/43000')
        
    f.close()
