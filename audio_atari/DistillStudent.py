import argparse
import os
import os.path as path
import json
import pickle
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_test import *
from baselines.common.trex_utils import preprocess
from keras.utils import to_categorical
import matplotlib.pyplot as plt

import librosa
from robotaxi.gameplay.openaiwrappers import make_openai_gym_environment
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import pickle

from preprocessing import generate_demos
from preprocessing import create_training_data

from DistillModel import TeacherNet, StudentNet



def create_training_data_prev(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length, env_name):
    #collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    num_demos = len(demonstrations)

    #fixed size snippets with progress prior
    for n in range(num_snippets):
        ti = 0
        tj = 0
        #only add trajectories that are different returns
        while(ti == tj):
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        #print(ti, tj)
        #create random snippets
        #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
        min_length = min(len(demonstrations[ti]), len(demonstrations[tj]))
        rand_length = np.random.randint(min_snippet_length, max_snippet_length)
        if ti < tj: #pick tj snippet to be later than ti
            ti_start = np.random.randint(min_length - rand_length + 1)
            #print(ti_start, len(demonstrations[tj]))
            tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - rand_length + 1)
        else: #ti is better so pick later snippet in ti
            tj_start = np.random.randint(min_length - rand_length + 1)
            #print(tj_start, len(demonstrations[ti]))
            ti_start = np.random.randint(tj_start, len(demonstrations[ti]) - rand_length + 1)
        #print("start", ti_start, tj_start)
        traj_i = demonstrations[ti][ti_start:ti_start+rand_length:2] #skip everyother framestack to reduce size
        traj_j = demonstrations[tj][tj_start:tj_start+rand_length:2]
    
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
        if ti > tj:
            label = 0
        else:
            label = 1
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)

    print("maximum traj length", max_traj_length)
    return training_obs, training_labels

def create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length, env_name, human_ann, random_ranking):
    #collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    num_demos = len(demonstrations)

    #fixed size snippets with progress prior
    for n in range(num_snippets):
        ti = 0
        tj = 0
        #only add trajectories that are different returns
        while ti == tj:
            #pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        #print(ti, tj)
        #create random snippets
        #find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
        min_length = min(len(demonstrations[ti]), len(demonstrations[tj]))
        rand_length = np.random.randint(min_snippet_length, max_snippet_length)

        mid_i = np.random.randint(min_length)
        mid_j = np.random.randint(min_length)

        while human_ann[ti][mid_i]['word'] == None:
            mid_i = np.random.randint(min_length)

        while human_ann[tj][mid_j]['word'] == None:
            mid_j = np.random.randint(min_length)
        
        ti_start = int(max(0, mid_i - (rand_length / 2)))
        tj_start = int(max(0, mid_j - (rand_length / 2)))
        ti_end   = int(min(len(demonstrations[ti]), mid_i + (rand_length / 2)))
        tj_end   = int(min(len(demonstrations[tj]), mid_j + (rand_length / 2)))

        # compute the number of yes and nos
        ann_i = human_ann[ti][ti_start:ti_end] #skip everyother framestack to reduce size
        ann_j = human_ann[tj][tj_start:tj_end]

        score_i = 0
        score_j = 0
        for l in ann_i:
            if l['word'] == 'yes' and l['conf'] > 0.0:
                score_i += 1
            if l['word'] == 'no'  and l['conf'] > 0.0:
                score_i -= 1

        for l in ann_j:
            if l['word'] == 'yes' and l['conf'] > 0.0:
                score_j += 1
            if l['word'] == 'no'  and l['conf'] > 0.0:
                score_j -= 1

        # compute this separately after duration 
        traj_i = demonstrations[ti][ti_start:ti_end:2] #skip everyother framestack to reduce size
        traj_j = demonstrations[tj][tj_start:tj_end:2]
    
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))

        # compute reward on just seed 0. PPO on seed 0,1,2

        # yes minus no duration
        # 10s of yes minus 5s of yes
        if score_i > score_j:
            label = 0
        else:
            label = 1

        if random_ranking:
            if np.random.rand(1)[0] > 0.5:
                label = 1
            else:
                label = 0

        training_obs.append((traj_i, traj_j))
        training_labels.append(label)

    print("maximum traj length", max_traj_length)
    return training_obs, training_labels




def learn_reward(teacher, student, optimizer, training_inputs, training_outputs, num_iter, l1_reg, checkpoint_dir):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    ce_loss = nn.CrossEntropyLoss()
    l1_loss = nn.L1Loss()
    kl_loss = nn.KLDivLoss()
    softmax = nn.Softmax()
    ALPHA = 0.9

    student.train()

    #print(training_data[0])
    cum_loss = 0.0
    training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels = zip(*training_data)
        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            #zero out gradient
            optimizer.zero_grad()

            #forward + backward + optimize
            t_outputs, t_abs_rewards = teacher.forward(traj_i, traj_j)
            s_outputs, s_abs_rewards = student.forward(traj_i, traj_j)

            t_weight = teacher.distill_weights()
            s_weight = student.distill_weights()

            t_outputs = t_outputs.unsqueeze(0)
            s_outputs = s_outputs.unsqueeze(0)

            #t_soft = softmax(t_outputs)
            #s_soft = softmax(s_outputs)

            #loss = loss_criterion(outputs, labels) + l1_reg * abs_rewards
            loss  = (1 - ALPHA) * ce_loss(s_outputs, labels)
            loss += (ALPHA) * (l1_loss(s_weight, t_weight) #+ kl_loss(s_soft, t_soft))
            loss.backward()
            optimizer.step()

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 500 == 499:
                #print(i)
                print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
                print(f"abs_rewards: {abs_rewards.item()}\n")
                cum_loss = 0.0
                torch.save(student.state_dict(), checkpoint_dir)
    print("finished training")



def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    #print(training_data[0])
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            #print(inputs)
            #print(labels)
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            #forward to get logits
            outputs, abs_return = reward_network.forward(traj_i, traj_j)
            #print(outputs)
            _, pred_label = torch.max(outputs,0)
            #print(pred_label)
            #print(label)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)






def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
            rewards_from_obs.append(r)
    return rewards_from_obs

def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))


if __name__=="__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--reward_model_path', default='', help="name and location for learned model params")
    parser.add_argument('-t', '--teacher', default='', help="Name and location for existing teacher model")
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--data_dir', help="where agc data is located, e.g. path to atari_v1/")
    parser.add_argument('-v', '--variation', default=0, type=int, help="Whether or not to sort the variations prior to rankings or during create_training_data (0,1)")

    args = parser.parse_args()
    env_name = args.env_name
    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
        agc_env_name =  "spaceinvaders"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
        agc_env_name = "mspacman"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
        agc_env_name = "pinball"
    elif env_name == "montezumarevenge":
        env_id = "MontezumaRevengeNoFrameskip-v4"
        agc_env_name = "revenge"
    elif env_name == "qbert":
        env_id = "QbertNoFrameskip-v4"
        agc_env_name = "qbert"
    elif env_name == "enduro":
        env_id = "EnduroNoFrameskip-v4"
        agc_env_name = "enduro"
    elif env_name == "seaquest":
        env_id = "SeaquestNoFrameskip-v4"
        agc_env_name = "seaquest"
    else:
        print("env_name not supported")
        sys.exit(1)
    print("trajectory path: " + str(args.data_dir + "/trajectories"))
    print("path existence:  " + str(os.path.exists(args.data_dir + "/trajectories")))

    if args.variation != 0 and args.variation != 1 and args.variation != 2:
        raise Exception("Variation not allowed")

    env_type = "atari"
    print(env_type)
    #set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    print("Training reward for", env_id)
    num_trajs = 0 
    num_snippets = 6000
    num_super_snippets = 0
    min_snippet_length = 50 #min length of trajectory for training comparison
    maximum_snippet_length = 100

    lr = 0.00005
    weight_decay = 0.0
    num_iter = 5 #num times through training data
    l1_reg=0.0
    stochastic = True

    env = make_vec_env(env_id, 'atari', 1, seed,
                       wrapper_kwargs={
                           'clip_rewards':False,
                           'episode_life':False,
                       })


    env = VecFrameStack(env, 4)

    data_dir = args.data_dir
    dataset = ds.AtariDataset(data_dir)
    demonstrations, learning_returns, human_ann, human_heatmap, pase, raw_audio = agc_demos.get_preprocessed_trajectories(agc_env_name, dataset, data_dir, env_name)

    demo_lengths = [len(d) for d in demonstrations]
    print("demo lengths", demo_lengths)
    max_snippet_length = min(np.min(demo_lengths), maximum_snippet_length)
    print("max snippet length", max_snippet_length)

    print(len(learning_returns))
    print(len(demonstrations))
    print([a[0] for a in zip(learning_returns, demonstrations)])

    # Sorting variant for audio trajectories #1
    if args.variation == 0:
        number_of_ann = [0] * len(demonstrations)
        for i in range(len(demonstrations)):
            traj = human_ann[i]
            for l in traj:
                if l['word'] == 'yes' and l['conf'] > 0.0:
                    number_of_ann[i] += 1
                if l['word'] == 'no' and l['conf'] > 0.0:
                    number_of_ann[i] -= 1
        demonstrations = [x for _, x in sorted(zip(number_of_ann, demonstrations), key=lambda pair: pair[0])]
    else:
        demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

    sorted_returns = sorted(learning_returns)
    print(sorted_returns)
    if args.variation == 0:
        training_obs, training_labels = create_training_data_prev(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length, env)
    else:
        training_obs, training_labels = create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length, env, human_ann, args.variation == 2)
    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))
    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    student = StudentNet()
    teacher = TeacherNet()
    student.to(device)
    teacher.to(device)
    import torch.optim as optim

    teacher.load_state_dict(torch.load(args.teacher))
    teacher.eval()
    optimizer = optim.Adam(student.parameters(),  lr=lr, weight_decay=weight_decay)
    learn_reward(teacher, student, optimizer, training_obs, training_labels, num_iter, l1_reg, args.reward_model_path)
    

    torch.save(student.state_dict(), args.reward_model_path)

    with torch.no_grad():
        pred_returns = [predict_traj_return(student, traj) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

    print("accuracy", calc_accuracy(student, training_obs, training_labels))


    print(f"Total time: {time.time() - start}")
