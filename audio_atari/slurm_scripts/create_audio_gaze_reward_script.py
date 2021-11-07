# env_names = [['seaquest','Seaquest'],['spaceinvaders','SpaceInvaders'],['mspacman','MsPacman'],['enduro','Enduro'],['montezumarevenge','MontezumaRevenge']]
# env = ['seaquest','Seaquest']
# env = ['spaceinvaders','SpaceInvaders']
env = ['mspacman','MsPacman']
# server = 'titans' #'dgx'
server = 'dgx'

# rl_mul, gaze_mul, audio_mul = '0.35', '0.05', '0.6'
# rl_mul, gaze_mul, audio_mul = '0.1', '0.45', '0.45'
# rl_mul, gaze_mul, audio_mul = '0.4', '0.2', '0.4'
# rl_mul, gaze_mul, audio_mul = '0.1', '0.8', '0.1'
rl_mul, gaze_mul, audio_mul = '0.9', '0.05', '0.05'

# extra, extra_option = '_rl_cgl', 'rl_cgl'
# extra, extra_option = '_rl_cal', 'rl_cal'
# extra, extra_option = '_cgl_cal', 'cgl_cal'
# extra, extra_option = '_rl_cgl_cal', 'rl_cgl_cal'
extra, extra_option = '_rl_cgl_cal_'+rl_mul+'-'+gaze_mul+'-'+audio_mul, 'rl_cgl_cal'


bash_file_name = 'reward/reward'+extra+'_'+env[0]+'.sh'
f = open(bash_file_name,'w')
f.write("#!/bin/bash\n\n")

f.write('#SBATCH --job-name reward'+extra+'_'+env[0]+'\n')
f.write('#SBATCH --output=logs/reward'+extra+'_'+env[0]+'_%j.out\n')
f.write('#SBATCH --error=logs/reward'+extra+'_'+env[0]+'_%j.err\n')
f.write('#SBATCH --mail-user=akankshasaran@utexas.edu\n')
f.write('#SBATCH --mail-type=END,FAIL,REQUEUE\n')
f.write('#SBATCH --partition '+server+'\n')
f.write('#SBATCH --nodes=1\n')
f.write('#SBATCH --ntasks-per-node=1\n')
f.write('#SBATCH --time 72:00:00\n')
f.write('#SBATCH --gres=gpu:1\n')
f.write('#SBATCH --mem=20G\n')
f.write('#SBATCH --cpus-per-task=4\n')

# f.write('python LearnAudioGazeRewardAGC.py --env_name '+env[0]+' --data_dir ./frames --reward_model_path ./multimodal_reward_models/'+env[0]+extra+' --loss_type '+extra_option)
f.write('python LearnAudioGazeRewardAGC.py --env_name '+env[0]+' --data_dir ./frames --reward_model_path ./multimodal_reward_models/'+env[0]+extra+\
    ' --loss_type '+extra_option+' --rl_mul '+rl_mul+' --gaze_mul '+gaze_mul+' --audio_mul '+audio_mul)


f.close()