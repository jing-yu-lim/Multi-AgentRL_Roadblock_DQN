import numpy as np
class Roadblock_Env():
    def __init__(self, left_type=0, right_type=1, observable=True):
        '''
        type == 0: prefers left_agent to go first

        type == 1: prefers right_agent to go first

        type == 2: prefers none

        left_type: the type of the left agent
        
        right_type: the type of the right agent

        state space: [left_pos, right_pos, left_type, right_type]
 
        '''
        self.left_type = left_type 
        self.right_type = right_type
        self.observable = observable
        self.state = np.array([0,0, left_type, right_type], dtype=np.float32) \
                        if observable == True else np.array([0,0], dtype=np.float32)
        self.left_ideal_state, self.right_ideal_state, self.idle_state, self.dead_state = np.copy(self.state), np.copy(self.state), np.copy(self.state),np.copy(self.state)
        self.left_ideal_state[0] = 1 
  
        self.right_ideal_state[1] = 1
        self.dead_state[0:2] = 1

    def reset(self):
        self.state = np.array([0,0, self.left_type, self.right_type], dtype=np.float32) if self.observable == True \
                     else np.array([0,0], dtype=np.float32)
        return np.copy(self.state)
    
    def is_terminal(self):
        if self.state[0] == 1 or self.state[1] ==1:
            return True
        return False

    def reward(self, type):
        rew=0


        
        if (self.state == self.left_ideal_state).all(): 
            if type == 0:
                rew = 0.5
            else: 
                rew = 0.4 #type == 2 assigned 0.4 no matter what
        
        elif (self.state == self.right_ideal_state).all():
            if type == 1:
                rew = 0.5
            else: 
                rew = 0.4

        elif (self.state == self.dead_state).all():
            rew = -1.0
        elif (self.state == self.idle_state).all():
            rew = -0.3
        
        return rew

    def step(self, left_action, right_action):
        if left_action == 1:
            self.state[0] = 1
        if right_action == 1:
            self.state[1]=1
        left_reward = self.reward(self.left_type)
        right_reward = self.reward(self.right_type)
        new_obs = self.state
        done = self.is_terminal()
        return new_obs, left_reward, right_reward, done
        
