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
import torch.optim as optim
from run_test import *

import os

from gaze.coverage import *
from tensorboardX import SummaryWriter
from audio.contrastive_loss import *

# TODO: add a different cal loss (auxiliary task of getting audio ranks correct)
# (use shrink and perturb), start with audio, then standard TREX (vice versa)


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


def create_CAL_training_data(demonstrations, audio, num_snippets):

    snippet_pairs = []

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

    #fixed size snippets
    for n in range(num_snippets):
        ti, tj = 0, 0
        label_len_i, label_len_j = 0,0

        #pick two random demonstrations (could be the same demo)
        while(label_len_i==0 or label_len_j==0):
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)

            # randomly pick yes/no
            label = np.random.randint(2)

            # randomly pick corresponding index pair from ti
            label_len_i, label_len_j = len(indices[label][ti]), len(indices[label][tj])

        snippet_id_i, snippet_id_j = np.random.randint(label_len_i), np.random.randint(label_len_j)
        ti_start, ti_stop = indices[label][ti][snippet_id_i][0], indices[label][ti][snippet_id_i][1]
        tj_start, tj_stop = indices[label][tj][snippet_id_j][0], indices[label][tj][snippet_id_j][1]


        left = np.random.randint(int(left_offset*fr))
        right = np.random.randint(int(right_offset*fr))
        ti_start = max(0,ti_start-left)
        ti_stop = min(ti_stop+right,len(demonstrations[ti]))
        ti_start = np.random.randint(ti_start, ti_stop)

        tj_start = max(0,tj_start-left)
        tj_stop = min(tj_stop+right,len(demonstrations[tj]))
        tj_start = np.random.randint(tj_start, tj_stop)

        #print("start", ti_start, tj_start)
        traj_i = demonstrations[ti][ti_start:ti_stop] 
        traj_j = demonstrations[tj][tj_start:tj_stop]
    
        snippet_pairs.append((traj_i,traj_j))

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
        # attn_map = norm_operator(torch.squeeze(conv_map))
        attn_map = norm_operator(conv_map)

        conv_map_traj.append(attn_map)
        conv_map_stacked = torch.stack(conv_map_traj)

        return sum_rewards, sum_abs_rewards, conv_map_stacked



    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i, conv_map_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j, conv_map_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j, conv_map_i, conv_map_j

    def forward_cal(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i, conv_map_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j, conv_map_j = self.cum_return(traj_j)
        return cum_r_i.unsqueeze(0).unsqueeze(0), cum_r_j.unsqueeze(0).unsqueeze(0)


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



def learn_reward(reward_network, optimizer, training_data, cal_training_data, num_iter, l1_reg, checkpoint_dir, env, \
    loss_type, rl_mul=0.5, gaze_mul=0.25, audio_mul=0.25):

    #check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    reward_path = checkpoint_dir+'/'+env+'.params'
    tb_dir = os.path.join(checkpoint_dir,'tb/')
    if not os.path.isdir(tb_dir):
        os.makedirs(tb_dir)
    writer = SummaryWriter(tb_dir)

    loss_criterion = nn.CrossEntropyLoss()
    #print(training_data[0])
    cum_loss = 0.0
    gaze_reg = 0.5
    gaze_scale = 0.001

    audio_reg = 0.5
    audio_scale = 0.1

    batch_size = 32
    num_batches = int(len(cal_training_data)/batch_size)
    cal_loss_criterion = ContrastiveLoss(batch_size)

    training_inputs, training_outputs, training_gaze = training_data
    assert(len(training_outputs)>num_batches)

    training_data = list(zip(training_inputs, training_outputs, training_gaze))
    k = 0
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        np.random.shuffle(cal_training_data)
        training_obs, training_labels, training_gaze = zip(*training_data)

        for i in range(len(training_labels)):

            k+=1
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            #zero out gradient
            optimizer.zero_grad()

            outputs, abs_rewards, conv_map_i, conv_map_j = reward_network.forward(traj_i, traj_j)
            outputs = outputs.unsqueeze(0)

            if 'rl' in loss_type:
                loss = loss_criterion(outputs, labels) + l1_reg * abs_rewards
                writer.add_scalar('ranking_loss', loss.item(), k)

            # gaze loss
            if 'cgl' in loss_type:
                gaze_loss = gaze_scale*CGL(training_gaze[i], conv_map_i, conv_map_j)
                writer.add_scalar('CGL', gaze_loss.item(), k)

            # audio loss
            if 'cal' in loss_type:
                # CAL
                j = i%num_batches
                rewards_m, rewards_n = torch.empty((0), dtype=torch.float, device = 'cuda'), torch.empty((0), dtype=torch.float, device = 'cuda')

                for m in range(j*batch_size,min((j+1)*batch_size,len(cal_training_data))):
                    traj_m, traj_n = cal_training_data[m]
                    
                    traj_m = np.array(traj_m)
                    traj_n = np.array(traj_n)
                    traj_m = torch.from_numpy(traj_m).float().to(device)
                    traj_n = torch.from_numpy(traj_n).float().to(device)
                    
                    r_m, r_n = reward_network.forward_cal(traj_m, traj_n)
                    
                    rewards_m = torch.cat((rewards_m,r_m),0)
                    rewards_n = torch.cat((rewards_n,r_n),0)


                audio_loss = audio_scale*cal_loss_criterion(rewards_m, rewards_n)
                writer.add_scalar('CAL', audio_loss.item(), k)

            if loss_type=='rl_cgl':
                loss = (1-gaze_reg)*loss + gaze_reg * gaze_loss
                writer.add_scalar('RL+CGL:', loss.item(), k)

            elif loss_type=='rl_cal':
                loss = (1-audio_reg)*loss + audio_reg * audio_loss
                writer.add_scalar('RL+CAL', loss.item(), k)

            elif loss_type=='cgl_cal':
                loss = (gaze_reg*gaze_loss) + (audio_reg*audio_loss)
                writer.add_scalar('CGL+CAL', loss.item(), k)

            elif loss_type=='rl_cgl_cal':
                # loss = (0.5)*loss + 0.25*audio_loss + 0.25*gaze_loss
                loss = (rl_mul*loss) + (audio_mul*audio_loss) + (gaze_mul*gaze_loss)
                writer.add_scalar('RL+CGL+CAL', loss.item(), k)

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
                torch.save(reward_net.state_dict(), reward_path)
    print("finished training")
    writer.close()


def lagrangian(loss1, loss2, eps, lamb):
    damping = 10.0
    scale = 0.001
    damp = damping * (eps-loss2.detach())
    return loss1 - (lamb-damp) * (eps-loss2*scale)


def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # loss_criterion = nn.CrossEntropyLoss()
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
            outputs, abs_return, _, _ = reward_network.forward(traj_i, traj_j)
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
    parser.add_argument('--num_snippets', default=6000, help="number of snippets to train the reward network with")
    parser.add_argument('--loss_type',default='rl_cgl_cal',type=str,help='losses to consider [rl, rl_cgl, rl_cal, cgl_cal, rl_cgl_cal]')
    parser.add_argument('--rl_mul', default=0.5, help="weight for ranking loss")
    parser.add_argument('--audio_mul', default=0.25, help="weight for audio loss")
    parser.add_argument('--gaze_mul', default=0.25, help="weight for gaze loss")

    # parser.add_argument('--gaze_loss_only', action='store_true')
    # parser.add_argument('--audio_loss_only', action='store_true')
    # parser.add_argument('--cgl_type',default='basic',type=str,help='way to combine cgl loss with pairwise ranking loss [basic, lagrangian]')

    args = parser.parse_args()
    # print(float(args.rl_mul)+float(args.gaze_mul)+float(args.audio_mul))
    assert(float(args.rl_mul)+float(args.gaze_mul)+float(args.audio_mul)==1.0)

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
    audio = [x for _, x in sorted(zip(learning_returns,human_ann), key=lambda pair: pair[0])]

    sorted_returns = sorted(learning_returns)
    print(sorted_returns)
    training_data = create_training_data(demonstrations, human_heatmap, num_snippets, min_snippet_length, max_snippet_length)
    training_obs, training_labels, training_gaze = training_data

    cal_training_data = create_CAL_training_data(demonstrations, audio, num_snippets)

    print("num training_obs", len(training_obs))
    print("num_labels", len(training_labels))
    # Now we create a reward network and optimize it using the training data.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net()
    reward_net.to(device)
    
    # if args.cgl_type=='basic' or args.gaze_loss_only:
    optimizer = optim.Adam(reward_net.parameters(),  lr=lr, weight_decay=weight_decay)
    learn_reward(reward_net, optimizer, training_data, cal_training_data, num_iter, l1_reg, args.reward_model_path, \
        env_name, args.loss_type, float(args.rl_mul), float(args.gaze_mul), float(args.audio_mul))
    # elif args.cgl_type=='lagrangian':
    #     optimizer = optim.SGD(reward_net.parameters(), lr=0.01)#, momentum=0.9)
    #     learn_reward_differential_combine(reward_net, optimizer, training_data, num_iter, l1_reg, args.reward_model_path, env_name)

    reward_path = args.reward_model_path+'/'+env_name+'.params'
    torch.save(reward_net.state_dict(), reward_path)

    with torch.no_grad():
        pred_returns = [predict_traj_return(reward_net, traj) for traj in demonstrations]
    for i, p in enumerate(pred_returns):
        print(i,p,sorted_returns[i])

    print("accuracy", calc_accuracy(reward_net, training_obs, training_labels))

    print(f"Total time: {time.time() - start}")