
# env_names = [['seaquest','Seaquest'],['spaceinvaders','SpaceInvaders'],['mspacman','MsPacman'],['enduro','Enduro'],['montezumarevenge','MontezumaRevenge']]
# env = ['seaquest','Seaquest']
# env = ['spaceinvaders','SpaceInvaders']
# env = ['mspacman','MsPacman']
env = ['enduro','Enduro']
# env = ['montezumarevenge','MontezumaRevenge']

server = 'titans' #'dgx'
seeds = ['0','1','2']

for seed in seeds:
    bash_file_name = 'eval/eval_'+env[0]+'_'+seed+'.sh'
    f = open(bash_file_name,'w')
    f.write("#!/bin/bash\n\n")

    f.write('#SBATCH --job-name eval_'+env[0]+'_'+seed+'\n')
    f.write('#SBATCH --output=logs/eval_'+env[0]+'_'+seed+'_%j.out\n')
    f.write('#SBATCH --error=logs/eval_'+env[0]+'_'+seed+'_%j.err\n')
    f.write('#SBATCH --mail-user=akankshasaran@utexas.edu\n')
    f.write('#SBATCH --mail-type=END,FAIL,REQUEUE\n')
    f.write('#SBATCH --partition '+server+'\n')
    f.write('#SBATCH --nodes=1\n')
    f.write('#SBATCH --ntasks-per-node=1\n')
    f.write('#SBATCH --time 72:00:00\n')
    f.write('#SBATCH --gres=gpu:1\n')
    f.write('#SBATCH --mem=10G\n')
    f.write('#SBATCH --cpus-per-task=4\n')

    f.write('python evaluateLearnedPolicy.py --env_name '+env[0]+' --checkpointpath ppo_models/'+env[0]+'_'+seed+'/checkpoints/43000')
        
    f.close()