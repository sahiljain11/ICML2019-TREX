import argparse
import agc_demos
import agc.dataset as ds

import pickle
from audio.contrastive_loss import ContrastiveSingleLoss, ContrastiveSinglePASELoss, ContrastiveSingleProsodyLoss, ContrastiveSinglePASEProsodyLoss
import gym
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from run_test import *

import os
import math
import torch.optim as optim

from audio.contrastive_loss import *
from tensorboardX import SummaryWriter


def create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length, env_name, audio):
    # TODO: update this function to use the audio snippets to rank trajectories instead of GT returns

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

        # compute the number of yes/no audio annotations
        audio_i = audio[ti][ti_start:ti_start+rand_length:2]
        audio_j = audio[ti][ti_start:ti_start+rand_length:2]
        yes_no_i = 0
        yes_no_j = 0

        for i in traj_i:
            i = i['word']
            if i == 'yes' or i == 'no':
                yes_no_i += 1

        for j in traj_j:
            j = j['word']
            if j == 'yes' or j == 'no':
                yes_no_j += 1

    
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
        #if ti > tj:
        if yes_no_i > yes_no_j:
            label = 0
        else:
            label = 1
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)

    print("maximum traj length", max_traj_length)
    return training_obs, training_labels

def create_CAL_training_data(demonstrations, audio, num_snippets):
    # sample yes and no utterances (frame IDs divisible by 16)
    # a vector of no snippets and a vector of all yes snippets
    # collect a list of negative and positive samples as vector 1
    # collect another list of negative and positive samples as vector 2
    # corresponding indices in vector1 and vector2 are positive pairs
    # then implement contrastive loss using code from https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/ 

    # length of snippet fixed between 1-5 seconds
    # ti_start and ti_stop determined from the yes/no labels
    # dont skip frame stacks (5 sec snippets at 60fps just give you ~20 frame stacks)
    # accumulate returns for these snippets

    # other option is to select snippets from yes, no pairs and directly add to the T-rex loss 
    # (returns for yes should be higher than return for no's)

    # fine tune with this small set

    snippet_pairs = []
    batch_size = 16

    # max_traj_length = 0
    # training_obs = []
    # training_labels = []
    num_demos = len(demonstrations)
    fr = math.ceil(60/16)
    conf = 0.7

    # in seconds
    left_offset, right_offset = 3.0, 3.0

    # for each demo, collect start and stop frame indices of yes and no indices (use conf = 0.8)
    # linear scan, yes start and stop boundary
    yes_indices, no_indices = [], []
    for j,demo in enumerate(audio):
        yes, no = [], []
        start = False
        for i,frame_stack in enumerate(demo):

            # start of a 'yes' speech segment
            if demo[i]['word']=='yes':
                if start==False and demo[i]['conf']>=conf:
                    if (i>0 and demo[i-1]['word']==None) or i==0:
                        start = True
                        start_idx = i
                    
                elif start==True and demo[i]['conf']>=conf:
                    if i==len(demo)-1 or (i<len(demo)-1 and demo[i+1]['word']==None):
                        start = False
                        stop_idx = i
                        yes.append([start_idx,stop_idx])

            # start of a 'no' speech segment
            elif demo[i]['word']=='no':
                if start==False and demo[i]['conf']>=conf:
                    if (i>0 and demo[i-1]['word']==None) or i==0:
                        start = True
                        start_idx = i
                    
                elif start==True and demo[i]['conf']>=conf:
                    if i==len(demo)-1 or (i<len(demo)-1 and demo[i+1]['word']==None):
                        start = False
                        stop_idx = i
                        no.append([start_idx,stop_idx])

        print(j, ', yes: ', len(yes), ' no: ', len(no))
        yes_indices.append(yes)
        no_indices.append(no)

    
    indices = [yes_indices, no_indices]
    print('yes utterances: ', len(yes_indices))
    print('no utterances: ', len(no_indices))

    num_batches = int(num_snippets/batch_size)

    #fixed size snippets
    for n in range(num_batches):
        ti, tj = 0, 0
        label_len_i, label_len_j = 0,0
    
        batch = []
        #pick two random demonstrations (could be the same demo)
        while(label_len_i==0 or label_len_j==0):
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)

            # randomly pick yes/no
            label = np.random.randint(2)
            neg_label = 1-label
            assert((neg_label==0 or neg_label==1) and neg_label!=label)

            label_len_i, label_len_j = len(indices[label][ti]), len(indices[label][tj])
            

        # randomly pick corresponding index pair from ti, tj
        snippet_id_i, snippet_id_j = np.random.randint(label_len_i), np.random.randint(label_len_j)
        ti_start, ti_stop = indices[label][ti][snippet_id_i][0], indices[label][ti][snippet_id_i][1]
        tj_start, tj_stop = indices[label][tj][snippet_id_j][0], indices[label][tj][snippet_id_j][1]

        left = np.random.randint(int(left_offset*fr))
        right = np.random.randint(int(right_offset*fr))
        ti_start_offset = max(0,ti_start-left)
        ti_stop_offset = min(ti_stop+right,len(demonstrations[ti]))

        # start point randomly selected from the left of the actual start
        ti_start = np.random.randint(ti_start_offset, ti_start+1)
        ti_stop = np.random.randint(ti_stop, ti_stop_offset+1)


        tj_start_offset = max(0,tj_start-left)
        tj_stop_offset = min(tj_stop+right,len(demonstrations[tj]))

        # start point randomly selected from the left of the actual start
        tj_start = np.random.randint(tj_start_offset, tj_start+1)
        tj_stop = np.random.randint(tj_stop, tj_stop_offset+1)

        #print("start", ti_start, tj_start)
        traj_i = demonstrations[ti][ti_start:ti_stop] 
        traj_j = demonstrations[tj][tj_start:tj_stop]
    
        batch.append((traj_i,traj_j))


        # separate out yes snippet pairs, no snippet pairs
        # create batches of data with one yes, 15 no pairs
        # create batches of data with one no, 15 yes pairs
        for k in range(batch_size-1):
            ti, tj = 0, 0
            neg_label_len_i, neg_label_len_j = 0,0
            # randomly select any demo pair
            while(neg_label_len_i==0 or neg_label_len_j==0):
                ti = np.random.randint(num_demos)
                tj = np.random.randint(num_demos)

                neg_label_len_i, neg_label_len_j = len(indices[neg_label][ti]), len(indices[neg_label][tj])


            # randomly pick corresponding index pair from ti, tj
            snippet_id_i, snippet_id_j = np.random.randint(neg_label_len_i), np.random.randint(neg_label_len_j)
            ti_start, ti_stop = indices[neg_label][ti][snippet_id_i][0], indices[neg_label][ti][snippet_id_i][1]
            tj_start, tj_stop = indices[neg_label][tj][snippet_id_j][0], indices[neg_label][tj][snippet_id_j][1]

            left = np.random.randint(int(left_offset*fr))
            right = np.random.randint(int(right_offset*fr))
            ti_start_offset = max(0,ti_start-left)
            ti_stop_offset = min(ti_stop+right,len(demonstrations[ti]))

            # start point randomly selected from the left of the actual start
            ti_start = np.random.randint(ti_start_offset, ti_start+1)
            # print(ti_stop,ti_stop_offset)
            ti_stop = np.random.randint(ti_stop, ti_stop_offset+1)


            tj_start_offset = max(0,tj_start-left)
            tj_stop_offset = min(tj_stop+right,len(demonstrations[tj]))

            # start point randomly selected from the left of the actual start
            tj_start = np.random.randint(tj_start_offset, tj_start+1)
            tj_stop = np.random.randint(tj_stop, tj_stop_offset+1)

            #print("start", ti_start, tj_start)
            traj_i = demonstrations[ti][ti_start:ti_stop] 
            traj_j = demonstrations[tj][tj_start:tj_stop]

            batch.append((traj_i,traj_j))


        snippet_pairs.append(batch)

    print('snippet pairs: ', len(snippet_pairs))
    
    return snippet_pairs

    

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
        x = F.leaky_relu(self.conv4(x))
        x = x.reshape(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)
        # print(r)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        # print(cum_r_i.item(), cum_r_j.item())
        # return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j
        return cum_r_i.unsqueeze(0).unsqueeze(0), cum_r_j.unsqueeze(0).unsqueeze(0)


def learn_reward(reward_network, optimizer, training_data, num_iter, l1_reg, checkpoint_path, env):
    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)
    # loss_criterion = nn.CrossEntropyLoss()
    #print(training_data[0])

    reward_path = checkpoint_path+'/'+env+'.params'
    tb_dir = os.path.join(checkpoint_path,'tb/')
    if not os.path.isdir(tb_dir):
        os.makedirs(tb_dir)
    writer = SummaryWriter(tb_dir)

    cum_loss = 0.0
    batch_size = 16
    num_batches = len(training_data)
    # num_batches = int(len(training_data)/batch_size)
    # training_data = list(zip(training_inputs, training_outputs))

    loss_criterion = ContrastiveSingleLoss(batch_size)
    k = 0
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        # training_obs, training_labels = zip(*training_data)

        # replace by batch size
        for j in range(num_batches):
            k+=1
            #zero out gradient
            optimizer.zero_grad()
            # print('batch: {}/{}'.format(j,num_batches))
            rewards_i, rewards_j = torch.empty((0), dtype=torch.float, device = 'cuda'), torch.empty((0), dtype=torch.float, device = 'cuda')
            # print('rewards_i shape: ',rewards_i.shape)
            # print('rewards_i size: ',rewards_i.nelement())

            # for i in range(j*batch_size,min((j+1)*batch_size,len(training_data))):
                # print('obs:',len(training_obs[i][0]))
                # print('data:',len(training_data[i][0]))

            # traj_i, traj_j will be a batch of size 16
            # loop through pairs in the batch for forward pass
            batch = training_data[j]
            for b in batch:
            # print(len(traj_i))
                # labels = np.array([training_labels[i]])
                ti, tj = b

                ti = np.array(ti)
                tj = np.array(tj)
                ti = torch.from_numpy(ti).float().to(device)
                tj = torch.from_numpy(tj).float().to(device)
                # labels = torch.from_numpy(labels).to(device)

                # forward + backward + optimize
                # outputs, abs_rewards = reward_network.forward(traj_i, traj_j)
                r_i, r_j = reward_network.forward(ti, tj)
                # print('r_i shape:',r_i.shape)

                # rewards_i.append([r_i.item()])
                # rewards_j.append([r_j.item()])
                rewards_i = torch.cat((rewards_i,r_i),0)
                rewards_j = torch.cat((rewards_j,r_j),0)

            # outputs = outputs.unsqueeze(0)

            # update loss to CAL for every batch
            # print(rewards_i.shape)
            loss = loss_criterion(rewards_i, rewards_j)
            print('loss: ',loss)
            writer.add_scalar('CAL', loss.item(), k)
            # loss = loss_criterion(outputs, labels) + l1_reg * abs_rewards
            loss.backward()
            optimizer.step()

        #print stats to see if learning
        item_loss = loss.item()
        cum_loss += item_loss
        if k % 500 == 499:
            print(k)
            print("epoch {}:{} loss {}".format(epoch,k,cum_loss))
            # print(f"abs_rewards: {abs_rewards.item()}\n")
            cum_loss = 0.0
            torch.save(reward_net.state_dict(), reward_path)
    print("finished training")
    writer.close()


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
    parser.add_argument('--num_snippets', default=6000, help="number of snippets to train the reward network with")
    parser.add_argument('--data_dir', help="where agc data is located, e.g. path to atari_v1/")

    # TODO: add Pairwise Ranking Loss on demo snippets which have audio
    # TODO: add option to finetune, train from scratch, or continue training from a pretrained network (shrink and perturb)

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
    num_snippets = int(args.num_snippets)
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
    demonstrations, learning_returns, human_ann, human_heatmap, pase, raw_audio  = agc_demos.get_preprocessed_trajectories(agc_env_name, dataset, data_dir, env_name)
    # print(human_ann)

    demo_lengths = [len(d) for d in demonstrations]
    print("demo lengths", demo_lengths)
    max_snippet_length = min(np.min(demo_lengths), maximum_snippet_length)
    print("max snippet length", max_snippet_length)

    print(len(learning_returns))
    print(len(demonstrations))
    print([a[0] for a in zip(learning_returns, demonstrations)])
    
    demonstrations = [x for _, x in sorted(zip(learning_returns,demonstrations), key=lambda pair: pair[0])]
    audio = [x for _, x in sorted(zip(learning_returns,human_ann), key=lambda pair: pair[0])]
    # gaze = [x for _, x in sorted(zip(learning_returns,human_heatmap), key=lambda pair: pair[0])]

    sorted_returns = sorted(learning_returns)
    print(sorted_returns)
    # training_obs, training_labels = create_training_data(demonstrations, num_trajs, num_snippets, min_snippet_length, max_snippet_length, env, audio)
    training_data = create_CAL_training_data(demonstrations, audio, num_snippets)
    
    print("num training_obs", len(training_data))
    # print("num_labels", len(training_labels))
    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net()
    reward_net.to(device)
    
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    learn_reward(reward_net, optimizer, training_data, num_iter, l1_reg, args.reward_model_path, env_name)

    reward_path = args.reward_model_path+'/'+env_name+'.params'
    torch.save(reward_net.state_dict(), reward_path)

    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

    # print("accuracy", calc_accuracy(reward_net, training_obs, training_labels))

    print(f"Total time: {time.time() - start}")