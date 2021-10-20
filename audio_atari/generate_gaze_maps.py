from gaze import gaze_heatmap as gh
from gaze import human_utils
from gaze import atari_head_dataset as ahd
import os
import cv2
import numpy as np
import os.path as path
# from agc_demos import *
import pickle as pkl
from scipy import misc
import argparse

class CreateGaze():
    def __init__(self, env_name, heatmap_shape=84):
        self.env_name = env_name
        self.heatmap_shape = heatmap_shape


    def StackFrames(self, frames):
        import copy
        """stack every four frames to make an observation (84,84,4)"""
        # stacked = np.zeros((len(frames), 84, 84, 4))
        stacked = []
        stacked_obs = np.zeros((84, 84, 4))
        for i in range(len(frames)):
            if i >= 3:
                stacked_obs[:, :, 0] = frames[i-3]
                stacked_obs[:, :, 1] = frames[i-2]
                stacked_obs[:, :, 2] = frames[i-1]
                stacked_obs[:, :, 3] = frames[i]
            else:
                stacked_obs[:, :, 0] = frames[i]
                stacked_obs[:, :, 1] = frames[i]
                stacked_obs[:, :, 2] = frames[i]
                stacked_obs[:, :, 3] = frames[i]
            stacked.append(np.expand_dims(copy.deepcopy(stacked_obs), 0))
            # stacked[i,:,:,:] = np.expand_dims(copy.deepcopy(stacked_obs), 0)
        # print(len(frames), len(stacked))
        return stacked


    def save_gaze_frames(self):

        model_path = 'gaze/pretrained_models/expert/' \
                    + self.env_name + '.hdf5'
        meanfile_path = 'gaze/pretrained_models/means/' + self.env_name + '.mean.npy'

        h = gh.PretrainedHeatmap(self.env_name, model_path, meanfile_path)

        # loop through appropriate frames/screens folder and subsequent demos inside it
        img_dir = os.path.join('frames/screens/',self.env_name)
        demo_dirs = os.listdir(img_dir)
        gaze_maps = {}
        for demo in demo_dirs:
            print(demo)
            demo_dir = os.path.join(img_dir,demo)
            if os.path.isdir(demo_dir):
                traj = []
                img_names = os.listdir(demo_dir)
                for img_name in img_names:
                    if img_name.endswith('.png'):
                        img_path = path.join(demo_dir,img_name) 
                        img = cv2.imread(img_path)
                        img_np = np.dot(img, [0.299, 0.587, 0.114]) # convert to grayscale
                        img_np = misc.imresize(img_np, [84, 84], interp='bilinear')
                        traj.append(img_np)

                stacked_traj = self.StackFrames(traj)
                # print(len(stacked_traj),stacked_traj[0].shape)  # 11480 (1, 84, 84, 4)

                gaze = h.get_heatmap(stacked_traj, self.heatmap_shape) 
                # print(gaze.shape)  # (11480, 84, 84)
                
                gaze_maps[int(demo)] = gaze
                

        # with open('gaze_'+self.env_name+'.pkl', 'wb') as handle:
        #     pkl.dump(gaze_maps, handle, protocol=pkl.HIGHEST_PROTOCOL)
        np.save('gaze_'+self.env_name+'.npy', gaze_maps) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--env', default='seaquest', help="AGC environment name: spaceinvaders, mspacman, montezumarevenge, seaquest, enduro")
    args = parser.parse_args()
    env_name = args.env

    g = CreateGaze(env_name)
    g.save_gaze_frames()

if __name__ == "__main__":
    main()      