import agc.dataset as ds
import agc.util as util
import numpy as np
from os import path, listdir
import cv2
cv2.ocl.setUseOpenCL(False)

import argparse
from baselines.common.trex_utils import preprocess

#need to grayscale and warp to 84x84
def GrayScaleWarpImage(image):
    """Warp frames to 84x84 as done in the Nature paper and later work."""
    width=84
    height=84
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    #frame = np.expand_dims(frame, -1)
    return frame

def MaxSkipAndWarpFrames(trajectory_dir, annotations, heatmaps, pase_vecs, raw_audio):
    """take a trajectory file of frames and max over every 3rd and 4th observation"""
    num_frames = len(listdir(trajectory_dir))
    skip=4

    sample_pic = np.random.choice(listdir(trajectory_dir))
    image_path = path.join(trajectory_dir, sample_pic)
    pic = cv2.imread(image_path)
    obs_buffer = np.zeros((2,)+pic.shape, dtype=np.uint8)
    max_frames = []
    anns = []
    maps = []
    pase = []
    raw  = []
    for i in range(num_frames):
        if i % skip == skip - 2:
            obs = cv2.imread(path.join(trajectory_dir, str(i) + ".png"))
            obs_buffer[0] = obs
        if i % skip == skip - 1:
            obs = cv2.imread(path.join(trajectory_dir, str(i) + ".png"))
            obs_buffer[1] = obs
            #warp max to 80x80 grayscale
            image = obs_buffer.max(axis=0)  # TODO: Choose the second frame's confidence and annotation
            # TODO: For gaze heatmaps, use average or pick one (say the second)
            warped = GrayScaleWarpImage(image)
            max_frames.append(warped)
            anns.append(annotations[i])
            #maps.append(heatmaps[i])
            pase.append(pase_vecs[i])
            raw.append(raw_audio[i])
    return max_frames, anns, maps, pase, raw

def StackFrames(frames, annotations, heatmaps, pase_vecs, raw_audio):
    import copy
    """stack every four frames to make an observation (84,84,4)"""
    stacked = []
    stacked_obs = np.zeros((84,84,4))
    anns = []
    maps = []
    pase = []
    raw  = []
    for i in range(len(frames)):
        if i >= 3:
            stacked_obs[:,:,0] = frames[i-3]
            stacked_obs[:,:,1] = frames[i-2]
            stacked_obs[:,:,2] = frames[i-1]
            stacked_obs[:,:,3] = frames[i]    # get audio annotation and heatmap for just this frame
            stacked.append(np.expand_dims(copy.deepcopy(stacked_obs),0))
            
            ann = annotations[i]
            if ann['word'] != None:
                for j in range(i-3, i):
                    if annotations[j]['word'] != None:
                        ann = annotations[j]
                        # print(ann)
                        break

            anns.append(ann)
            #maps.append(heatmaps[i])
            pase.append(pase_vecs[i])
            raw.append(raw_audio[i])
            #assert len(raw[i-3]) != 0

    print(f"StackFrame length: {len(raw)}")
    return stacked, anns, maps, pase, raw



def get_sorted_traj_indices(env_name, dataset):
    #need to pick out a subset of demonstrations based on desired performance
    #first let's sort the demos by performance, we can use the trajectory number to index into the demos so just
    #need to sort indices based on 'score'
    g = env_name
    #Note, we're only keeping the full demonstrations that end in terminal to avoid people who quit before the game was over
    traj_indices = []
    traj_scores  = []
    traj_ann     = []
    traj_heatmap = []
    # print(dataset.trajectories.keys())
    for t in dataset.trajectories[g]:
        if env_name == "revenge":
            traj_indices.append(t)
            traj_scores.append(dataset.trajectories[g][t][-1]['score'])
            traj_ann.append(dataset.trajectories[g][t][-1]['ann'])
            traj_heatmap.append(dataset.trajectories[g][t][-1]['heatmap'])
            

        elif dataset.trajectories[g][t][-1]['terminal']:
            traj_indices.append(t)
            traj_scores.append(dataset.trajectories[g][t][-1]['score'])
            traj_ann.append(dataset.trajectories[g][t][-1]['ann'])
            traj_heatmap.append(dataset.trajectories[g][t][-1]['heatmap'])

    annotations = dataset.annotations[g]
    heatmaps    = dataset.heatmap[g]
    pase        = dataset.pasevec[g]
    raw_audio   = dataset.raw_audio[g]

    #for k in dataset.annotations[g].keys():
        #assert len(dataset.annotations[g][k]) == len(dataset.pasevec[g][k])
        #assert len(dataset.annotations[g][k]) == len(dataset.raw_audio[g][k])
        #assert len(dataset.annotations[g][k]) == len(dataset.heatmaps[g][k])

    sorted_traj_indices = [x for _, x in sorted(zip(traj_scores, traj_indices), key=lambda pair: pair[0])]
    sorted_traj_scores = sorted(traj_scores)

    print(sorted_traj_scores)
    print(sorted_traj_indices)
    print("Max human score", max(sorted_traj_scores))
    print("Min human score", min(sorted_traj_scores))

    sorted_traj_ann     = []
    sorted_traj_heatmap = []
    for i in sorted_traj_indices:
        sorted_traj_ann.append(traj_ann[i-1])
        sorted_traj_heatmap.append(traj_heatmap[i-1])
    
    seen_scores = set()
    non_duplicates = []
    for i,s,a,h in zip(sorted_traj_indices, sorted_traj_scores, sorted_traj_ann, sorted_traj_heatmap):
        if s not in seen_scores:
            seen_scores.add(s)
            non_duplicates.append((i,s,a,h))
    print("num non duplicate scores", len(seen_scores))
    #num_demos = 12
    #if env_name == "spaceinvaders":
    #    start = 0
    #    skip = 4
    #elif env_name == "revenge":
    #    start = 0
    #    skip = 1
    #elif env_name == "qbert":
    #    start = 0
    #    skip = 3
    #elif env_name == "mspacman":
    #    start = 0
    #    skip = 3
    #elif env_name == "pinball":
    #    start = 0
    #    skip = 1
    ## TODO: Double check skip values
    #elif env_name == "seaquest":
    #    start = 0
    #    skip = 3
    #elif env_name == "enduro":
    #    start = 0
    #    skip = 3

    #demos = non_duplicates[start:num_demos*skip + start:skip]
    #print("(index, score) pairs:", len(demos))
    #return demos
    print("(index, score) pairs:", len(non_duplicates))
    return non_duplicates, annotations, heatmaps, pase, raw_audio

def get_preprocessed_trajectories(env_name, dataset, data_dir, preprocess_name):
    """returns an array of trajectories corresponding to what you would get running checkpoints from PPO
       demonstrations are grayscaled, maxpooled, stacks of 4 with normalized values between 0 and 1 and
       top section of screen is masked
    """

   
    print("generating human demos for", env_name)
    demos, annotations, heatmaps, pase, raw_audio = get_sorted_traj_indices(env_name, dataset)

    human_scores   = []
    human_demos    = []
    human_ann      = []
    human_heatmap  = []
    human_pase     = []
    human_audio    = []
    for indx, score, _, _ in demos:
        human_scores.append(score)
        traj_dir = path.join(data_dir, 'screens', env_name, str(indx))
        #print("generating traj from", traj_dir)
        maxed_traj, filtered_ann, filtered_heatmap, filtered_pase, filtered_raw_audio = MaxSkipAndWarpFrames(traj_dir, annotations[indx], heatmaps[indx], pase[indx], raw_audio[indx])
        stacked_traj, filtered_ann, filtered_heatmap, filtered_pase, filtered_raw_audio = StackFrames(maxed_traj, filtered_ann, filtered_heatmap, filtered_pase, filtered_raw_audio)
        demo_norm_mask = []
        #normalize values to be between 0 and 1 and have top part masked
        for ob in stacked_traj:
            demo_norm_mask.append(preprocess(ob, preprocess_name)[0])
        human_demos.append(demo_norm_mask)
        human_ann.append(filtered_ann)
        human_heatmap.append(filtered_heatmap)
        human_pase.append(filtered_pase)
        human_audio.append(filtered_raw_audio)
    return human_demos, human_scores, human_ann, human_heatmap, human_pase, human_audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--agc_env', default='', help="AGC environment name: spaceinvaders, qbert, mspacman, revenge, pinball")
    parser.add_argument('--atari_full_env', help="Full lowercase atari env name: eg spaceinvadrs, montezumarevenge, etc")
    parser.add_argument('--datadir', default=None, help='location of atari gc data')
    args = parser.parse_args()
    env_name = args.agc_env
    args.datadir
    data_dir = args.datadir
    dataset = ds.AtariDataset(data_dir)
    human_demos, human_scores, human_ann, human_heatmap = get_preprocessed_trajectories(env_name, dataset, data_dir, args.atari_full_env)
    print(human_scores)
    
    print("max", max(human_scores))
    print("mean", np.mean(human_scores))
