import warnings
warnings.filterwarnings('ignore')

import librosa
import os
import torch
import json
import numpy as np
from numpy import linalg as LA
import argparse
from progress.bar import Bar

with open('complete.json') as f:
    complete = json.load(f)
with open('gamekeys.json') as f:
    mapping  = json.load(f)


def path_helper(path_string: str) -> str:
    return os.path.join(os.getcwd(), path_string)

def frames_finder(wav_file: str, game: str):
    name = wav_file.split(".")[0]
    traj_name = mapping[game][name]
    if 'R0YWVY6RKQ' in traj_name:
        return -1, traj_name
    return len(complete[traj_name].keys()), traj_name


def pase_on_file(data_path: str, num_frames: int, name: str):
    data = [None] * num_frames
    raw  = [None] * num_frames

    y, sr = librosa.load(data_path, sr=48000)
    SECONDS = librosa.get_duration(y=y, sr=sr)
    y = torch.tensor(y).view((1, 1, -1))
    CONST = 16
    AUDIO_FRAME = y.shape[2]

    #sec_per_frame = SECONDS / num_frames
    frame_divide_16 = num_frames / CONST
    divided = int(AUDIO_FRAME / frame_divide_16)

    bar = Bar(f'Traj #{name}', max=num_frames)
    for i in range(num_frames):
        if i % CONST == 0:
            #start = max(i * sec_per_frame - 0.5, 0.0)
            #y, sr = librosa.load(data_path, offset=start, duration=1.0, sr=48000)
            num = i / CONST
            start = int(divided * (num))
            end   = int(divided * (num + 1))

            s = y[:,:,start:end]
            #result = pase(s, device="cpu")[:,:,-1].view((256))

            #temp = result.detach().numpy()
            temp = 0
            temp2 = s

        data[i] = temp
        raw[i]  = temp2
        bar.next()
    bar.finish()

    return data, raw

if __name__ == "__main__":
    STR = './frames/human_audio'
    count = 0
    for game in sorted(os.listdir(STR)):
        print(f"\nStarting {game}")
        final = {}
        raw   = {}
        for wav_file in os.listdir(f'{STR}/{game}/'):
            wav_path = path_helper(f'{STR}/{game}/{wav_file}')
            (num_frames, traj_name) = frames_finder(wav_file, game)

            if num_frames == -1:
                continue

            (data, r) = pase_on_file(wav_path, num_frames, str(count))
            num = int(wav_file.split('.')[0])

            final[num] = data
            raw[traj_name] = r

            count += 1

        print(f"\nSaving {game}")
        #with open(f'{game}_vec.npy', 'wb') as f:
        #    np.save(f, final)
        with open(f'./frames/pase_raw/{game}_raw.npy', 'wb') as f:
            np.save(f, raw)