
# env_names = [['seaquest','Seaquest'],['spaceinvaders','SpaceInvaders'],['mspacman','MsPacman'],['enduro','Enduro'],['montezumarevenge','MontezumaRevenge']]
# env = ['seaquest','Seaquest']
env = ['spaceinvaders','SpaceInvaders']
# env = ['mspacman','MsPacman']
server = 'titans' #'dgx'

bash_file_name = 'reward/reward_CAL_'+env[0]+'.sh'
f = open(bash_file_name,'w')
f.write("#!/bin/bash\n\n")

f.write('#SBATCH --job-name reward_'+env[0]+'\n')
f.write('#SBATCH --output=logs/reward_'+env[0]+'_%j.out\n')
f.write('#SBATCH --error=logs/reward_'+env[0]+'_%j.err\n')
f.write('#SBATCH --mail-user=akankshasaran@utexas.edu\n')
f.write('#SBATCH --mail-type=END,FAIL,REQUEUE\n')
f.write('#SBATCH --partition '+server+'\n')
f.write('#SBATCH --nodes=1\n')
f.write('#SBATCH --ntasks-per-node=1\n')
f.write('#SBATCH --time 72:00:00\n')
f.write('#SBATCH --gres=gpu:1\n')
f.write('#SBATCH --mem=20G\n')
f.write('#SBATCH --cpus-per-task=4\n')

f.write('python LearnAudioRewardAGC.py --env_name '+env[0]+' --data_dir ./frames --reward_model_path ./multimodal_reward_models/'+env[0]+'_CAL_only')
      
f.close()