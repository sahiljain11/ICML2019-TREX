import argparse
import agc_demos
import agc.dataset as ds

import pickle
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_test import *

import os

from gaze.coverage import *

"""
python LearnAtariRewardAGC.py --env_name mspacman --data_dir /home/sahilj/forked/ICML2019-TREX/audio_atari/frames --reward_model_path ./learned_models/mspacman.params
python LearnAtariRewardAGC.py --env_name spaceinvaders --data_dir /home/sahilj/forked/ICML2019-TREX/audio_atari/frames --reward_model_path ./learned_models/spaceinvaders0.params --seed 0
"""

'''
To train on a reward network, use the following command:
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=/home/sahilj/forked/ICML2019-TREX/tflogs python -m baselines.run --alg=ppo2 --env=MsPacmanNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/mspacman.params --seed 0 --num_timesteps=5e7 --save_interval=500 --num_env 9

OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=/home/sahilj/forked/ICML2019-TREX/tflogs_space00 python -m baselines.run --alg=ppo2 --env=SpaceInvadersNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/spaceinvaders0.params --seed 0 --num_timesteps=5e7 --save_interval=500 --num_env 9
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=/home/sahilj/forked/ICML2019-TREX/tflogs_space01 python -m baselines.run --alg=ppo2 --env=SpaceInvadersNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/spaceinvaders0.params --seed 1 --num_timesteps=5e7 --save_interval=500 --num_env 9
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=/home/sahilj/forked/ICML2019-TREX/tflogs_space10 python -m baselines.run --alg=ppo2 --env=SpaceInvadersNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/spaceinvaders1.params --seed 0 --num_timesteps=5e7 --save_interval=500 --num_env 9
OPENAI_LOG_FORMAT='stdout,log,csv,tensorboard' OPENAI_LOGDIR=/home/sahilj/forked/ICML2019-TREX/tflogs_space11 python -m baselines.run --alg=ppo2 --env=SpaceInvadersNoFrameskip-v4 --custom_reward pytorch --custom_reward_path learned_models/spaceinvaders1.params --seed 1 --num_timesteps=5e7 --save_interval=500 --num_env 9
'''

def create_training_data(demonstrations, gaze_maps, num_snippets, min_snippet_length, max_snippet_length):
    #collect training data
    max_traj_length = 0
    training_obs, training_labels, training_gaze = [], [], []
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

        gaze_i = gaze_maps[ti][ti_start:ti_start+rand_length:2]
        gaze_j = gaze_maps[ti][ti_start:ti_start+rand_length:2]

        # print(gaze_i)
    
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
        if ti > tj:
            label = 0
        else:
            label = 1
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        training_gaze.append((gaze_i, gaze_j))

    print("maximum traj length", max_traj_length)
    return training_obs, training_labels, training_gaze



class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)



    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        x = traj.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x4 = F.leaky_relu(self.conv4(x))
        x = x4.reshape(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))

        # prepare conv map to be returned for gaze loss
        conv_map_traj = []
        conv_map_stacked = torch.tensor([[]]) # 26,11,9,7 (size of conv layers)

        gaze_conv = x4
        conv_map = gaze_conv

        # 1x1 convolution followed by softmax to get collapsed and normalized conv output
        norm_operator = nn.Conv2d(16, 1, kernel_size=1, stride=1)
        if torch.cuda.is_available():
            norm_operator.cuda()
        attn_map = norm_operator(torch.squeeze(conv_map))

        conv_map_traj.append(attn_map)
        conv_map_stacked = torch.stack(conv_map_traj)

        return sum_rewards, sum_abs_rewards, conv_map_stacked



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i, conv_map_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j, conv_map_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j, conv_map_i, conv_map_j


def CGL(training_gaze, conv_map_i, conv_map_j):
    gaze_i, gaze_j = training_gaze
    gaze_i = torch.unsqueeze(torch.squeeze(torch.tensor(gaze_i, device=device)),1).float() # list of torch tensors
    gaze_j = torch.unsqueeze(torch.squeeze(torch.tensor(gaze_j, device=device)),1).float()
    # print('gaze:',gaze_i.shape)

    attn_map_i = torch.unsqueeze(torch.squeeze(conv_map_i),1)
    attn_map_j = torch.unsqueeze(torch.squeeze(conv_map_j),1)

    # print('conv map:', attn_map_i.shape)

    attn_map_i = F.interpolate(attn_map_i, (84,84), mode="bilinear", align_corners=False)
    attn_map_j = F.interpolate(attn_map_j, (84,84), mode="bilinear", align_corners=False)

    kl_i = computeKL_batch(attn_map_i, gaze_i)
    kl_j = computeKL_batch(attn_map_j, gaze_j)
    gaze_loss_i = torch.sum(kl_i)
    gaze_loss_j = torch.sum(kl_j)

    gaze_loss_total = (gaze_loss_i + gaze_loss_j)
    return gaze_loss_total



def learn_reward(reward_network, optimizer, training_data, num_iter, l1_reg, checkpoint_dir):

    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    loss_criterion = nn.CrossEntropyLoss()
    #print(training_data[0])
    cum_loss = 0.0
    gaze_reg = 0.5

    training_inputs, training_outputs, training_gaze = training_data

    training_data = list(zip(training_inputs, training_outputs, training_gaze))

    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels, training_gaze = zip(*training_data)

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
            outputs, abs_rewards, conv_map_i, conv_map_j = reward_network.forward(traj_i, traj_j)
            #print(outputs[0], outputs[1])
            #print(labels.item())
            outputs = outputs.unsqueeze(0)
            #print("outputs", outputs)
            #print("labels", labels)
            loss = loss_criterion(outputs, labels) + l1_reg * abs_rewards

            # gaze loss
            gaze_loss = CGL(training_gaze[i], conv_map_i, conv_map_j)
            loss = (1-gaze_reg)*loss + gaze_reg * gaze_loss

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
                torch.save(reward_net.state_dict(), checkpoint_dir)
    print("finished training")




def lagrangian(loss1, loss2, eps, lamb):
    damping = 10.0
    damp = damping * (eps-loss2.detach())
    return loss1 - (lamb-damp) * (eps-loss2)

def learn_reward_differential_combine(reward_network, optimizer, training_data, num_iter, l1_reg, checkpoint_dir):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    loss_criterion = nn.CrossEntropyLoss()
    #print(training_data[0])
    cum_loss = 0.0
    gaze_reg = 0.5

    training_inputs, training_outputs, training_gaze = training_data
    training_data = list(zip(training_inputs, training_outputs, training_gaze))

    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs, training_labels, training_gaze = zip(*training_data)

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
            outputs, abs_rewards, conv_map_i, conv_map_j = reward_network.forward(traj_i, traj_j)
            #print(outputs[0], outputs[1])
            #print(labels.item())
            outputs = outputs.unsqueeze(0)
            #print("outputs", outputs)
            #print("labels", labels)
            loss = loss_criterion(outputs, labels) + l1_reg * abs_rewards

            # gaze loss
            gaze_loss = CGL(training_gaze[i], conv_map_i, conv_map_j)
            lamb = 0.0
            eps = 0.7
            # loss = (1-gaze_reg)*loss + gaze_reg * gaze_loss
            loss = lagrangian(loss, gaze_loss, eps, lamb)

            loss.backward()
            optimizer.step()

            # check if lambda is updated by step function
            print(lamb)

            #print stats to see if learning
            item_loss = loss.item()
            cum_loss += item_loss
            if i % 500 == 499:
                #print(i)
                print("epoch {}:{} loss {}".format(epoch,i, cum_loss))
                print(f"abs_rewards: {abs_rewards.item()}\n")
                cum_loss = 0.0
                torch.save(reward_net.state_dict(), checkpoint_dir)

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
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--data_dir', help="where agc data is located, e.g. path to atari_v1/")

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
    demonstrations, learning_returns, human_ann, human_heatmap = agc_demos.get_preprocessed_trajectories(agc_env_name, dataset, data_dir, env_name)
    # print(human_ann)

    demo_lengths = [len(d) for d in demonstrations]
    print("demo lengths", demo_lengths)
    max_snippet_length = min(np.min(demo_lengths), maximum_snippet_length)
    print("max snippet length", max_snippet_length)

    print(len(learning_returns))
    print(len(demonstrations))
    print([a[0] for a in zip(learning_returns, demonstrations)])
    
    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]

    sorted_returns = sorted(learning_returns)
    print(sorted_returns)
    training_data = create_training_data(demonstrations, human_heatmap, num_snippets, min_snippet_length, max_snippet_length)
    training_obs, training_labels, training_gaze = training_data

    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))
    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net()
    reward_net.to(device)
    import torch.optim as optim
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    learn_reward(reward_net, optimizer, training_data, num_iter, l1_reg, args.reward_model_path)
    

    torch.save(reward_net.state_dict(), args.reward_model_path)

    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

    print("accuracy", calc_accuracy(reward_net, training_obs, training_labels))


    print(f"Total time: {time.time() - start}")
