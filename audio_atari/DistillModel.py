import torch
import torch.nn as nn
import torch.nn.functional as F

class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)

        self.fc3 = nn.Linear(3, 8)
        self.fc4 = nn.Linear(8, 16)

        self.combined1 = nn.Linear(48, 24)
        self.combined2 = nn.Linear(24, 1)


    def cum_return(self, traj, human_ann):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0
        x1 = traj.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network
        x1 = F.leaky_relu(self.conv1(x1))
        x1 = F.leaky_relu(self.conv2(x1))
        x1 = F.leaky_relu(self.conv3(x1))
        x1 = F.leaky_relu(self.conv4(x1))
        x1 = x1.reshape(-1, 784)
        x1 = F.leaky_relu(self.fc1(x1))
        x1 = F.leaky_relu(self.fc2(x1))

        x2 = F.leaky_relu(self.fc3(human_ann))
        x2 = F.leaky_relu(self.fc4(x2))
        x2 = x2.reshape(-1, 16)

        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.combined1(x))
        r = self.combined2(x)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards

    def forward(self, traj_i, traj_j, human_i, human_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i, human_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j, human_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j

    def distill_weights(self):
        return self.combined2.weight



class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 64 input (8x8 grid) x 7 vector
        self.fc1 = nn.Linear(448, 70) # should be 8x8x7
        #self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(70, 16)
        self.fc4 = nn.Linear(16, 1)

        # 1 output (either better or worse)

    def cum_return(self, traj):
        '''calculate cumulative return of trajectory'''
        sum_rewards = 0
        sum_abs_rewards = 0

        # pass through state layers
        x = F.leaky_relu(self.fc1(traj))
        #x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        r = self.fc4(x)

        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards

    def forward(self, traj_i, traj_j):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)),0), abs_r_i + abs_r_j, self.fc4.weight

    def distill_weights(self):
        return self.fc1.weight