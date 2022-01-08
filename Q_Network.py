from math import gamma
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super().__init__()
        self.input_dims=input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1=nn.Linear(*self.input_dims,self.fc1_dims)  #unpacking is to facilitate inputs from conv net
        self.fc2=nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3=nn.Linear(self.fc2_dims,n_actions)         #output a q value for each action
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.99)
        self.loss=nn.MSELoss()
        self.device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions  = self.fc3(x)
        return actions