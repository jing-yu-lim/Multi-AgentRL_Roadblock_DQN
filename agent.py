#https://www.youtube.com/watch?v=wc-FxNENg9U&t=961s

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from Q_Network import DeepQNetwork

class Agent():
    """
    gamma: discount factor

    epsilon: explore vs exploit
    
    eps_min: min epsilon value

    eps_dec: how much decrement to epsilon at each step

    """
    def __init__ (self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                max_mem_size = 100000, eps_min = 0.01, eps_dec=5e-4):
        self.gamma = gamma 
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]                           #makes it easy to select an action at random
        self.mem_size=max_mem_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, 
                                fc1_dims=256, fc2_dims=256)

        self.Q_target = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, 
                                fc1_dims=256, fc2_dims=256)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_target.eval()

        self.state_memory = np.zeros((self.mem_size, *input_dims),dtype = np.float32) #dtype is important
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                        dtype = np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)                 #value of terminal state is 0; pass in done flag

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index]= state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action (self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item() #action encoded as an integer
        else:
            action = np.random.choice(self.action_space)
        
        return action

    def learn(self):            #start learning as soon as memory filled up to batch size
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace = False)

        batch_index = np.arange(self.batch_size,dtype = np.int32) #slice each batch

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device) #batch size x input_dims
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch] #slice by batch number, and integer-encoded action
        q_next = self.Q_target.forward(new_state_batch)

        q_next[terminal_batch]=0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0] #index 0 because max gives both value and index
        loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min




