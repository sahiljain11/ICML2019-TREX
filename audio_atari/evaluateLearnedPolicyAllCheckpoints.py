import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import sys
import pickle
import gym
import time
import numpy as np
import random
import torch
from run_test import *
import string
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
"""
taskset -c 1-60 nice -n 20 python evaluateLearnedPolicyAllCheckpoints.py --env_name seaquest -l /home/sahilj/forked/ICML2019-TREX/tflogs_seaquest12 -n SentimentRanking1_Seed2
"""

def evaluate_learned_policy(env_name, checkpointpath):
    with tf.Session(graph=tf.Graph()):
        if env_name == "spaceinvaders":
            env_id = "SpaceInvadersNoFrameskip-v4"
        elif env_name == "mspacman":
            env_id = "MsPacmanNoFrameskip-v4"
        elif env_name == "videopinball":
            env_id = "VideoPinballNoFrameskip-v4"
        elif env_name == "beamrider":
            env_id = "BeamRiderNoFrameskip-v4"
        elif env_name == "montezumarevenge":
            env_id = "MontezumaRevengeNoFrameskip-v4"
        else:
            env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"

        env_type = "atari"

        stochastic = True

        #env id, env type, num envs, and seed
        env = make_vec_env(env_id, 'atari', 1, 0,
                           wrapper_kwargs={
                               'clip_rewards':False,
                               'episode_life':False,
                           })



        env = VecFrameStack(env, 4)

        agent = PPO2Agent(env, env_type, stochastic)  #defaults to stochastic = False (deterministic policy)
        #agent = RandomAgent(env.action_space)

        learning_returns = []
        print(checkpointpath)

        agent.load(checkpointpath)
        episode_count = 30
        for i in range(episode_count):
            done = False
            traj = []
            r = 0

            ob = env.reset()
            #traj.append(ob)
            #print(ob.shape)
            steps = 0
            acc_reward = 0
            while True:
                action = agent.act(ob, r, done)
                #print(action)
                ob, r, done, _ = env.step(action)

                #print(ob.shape)
                steps += 1
                #print(steps)
                acc_reward += r[0]
                if done:
                    print("steps: {}, return: {}".format(steps,acc_reward))
                    break
            learning_returns.append(acc_reward)



        env.close()

    return learning_returns

    #return([np.max(learning_returns), np.min(learning_returns), np.mean(learning_returns)])

def already_computed(res):
    f = open(res)

    returns = []
    for l in f.readlines():
        returns.append(float(l.strip('\n')))
    return returns

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=1234, help="random seed for experiments")
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('-l', '--pathlist', action='append', help='path to checkpoints to run eval on')
    parser.add_argument('-n', '--listnames', action='append', help='names for each checkpoint to be labeled')
    args = parser.parse_args()
    env_name = args.env_name

    #set seeds
    #seed = int(args.seed)
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #tf.compat.v1.set_random_seed(seed)

    colors = ['blue', 'orange', 'green', 'purple', 'brown', 'red']
    c = -1
    all_variations = args.listnames

    for l in args.pathlist:
        print("*"*10)
        c += 1
        all_x    = []
        all_mean = []
        all_std  = []

        #variation = l.split(f'tflogs_small_{env_name}')[1]
        #variation = l.split(f'{env_name}_')[1]
        #all_variations.append(variation)
        variation = all_variations[c]

        complete_path = os.path.join(l, 'checkpoints')

        for name in sorted(os.listdir(complete_path)):
            num = int(name)
            #print(num)

            checkpointpath = os.path.join(complete_path, name)
            all_x.append(num)
            #print("*"*10)
            #print(env_name)
            #print("*"*10)

            res = "./eval/" + env_name + '_' + checkpointpath.replace("/","_") + "_evaluation.txt"

            # see if returns has already been computed
            if os.path.exists(res):
                #print(f'Cached: {checkpointpath}')
                returns = already_computed(res)
                val = sum(returns)
            else:
                returns = evaluate_learned_policy(env_name, checkpointpath)
                #write returns to file

                val = 0.0
                f = open("./eval/" + env_name + '_'+ checkpointpath.replace("/","_") + "_evaluation.txt",'w')
                #f = open(f"./eval/{res}.txt", "w")
                for r in returns:
                    f.write("{}\n".format(r))
                    val += r
                f.close()

            val = val / len(returns)
            err = np.std(returns)/np.sqrt(len(returns))

            #print(f"Stored results in {res}.txt")

            all_mean.append(val)
            all_std.append(err)

        print(complete_path)
        print(f"Average: {val}")
        print(f"Std err: {err}")
        x = np.array(all_x)
        y = np.array(all_mean)
        e = np.absolute(np.array(all_std))

        plt.plot(x, y, label=variation, color=colors[c])
        plt.fill_between(x, y - e, y + e, alpha=0.5, facecolor=colors[c])
    print("*"*10)

    name = ''
    for i in range(0, len(all_variations)):
        if i == 0:
            name += f'{all_variations[i]}'
        else:
            name += f'_{all_variations[i]}'

    plt.title(f'{env_name} {name}')
    plt.xlabel('PPO Checkpoints')
    plt.ylabel('PPO Rewards')
    plt.legend(all_variations)

    plt.savefig(f'./images/{env_name}{name}.png')

    print(f'Stored image in ./images/{env_name}{name}.png')

