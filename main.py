from agent import Agent
from utils import plotLearning
from environment import Roadblock_Env
import numpy as np
import os
if __name__ == '__main__':
    env = Roadblock_Env()
    left_agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 64, n_actions=2, eps_min=0.01, input_dims=[4],lr = 0.003)
    right_agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 64, n_actions=2, eps_min=0.01, input_dims=[4],lr = 0.003)
    
    scores, eps_history = [], [] 
    n_games =50
    target_update=100 #how often to update target network
    lr_step=50       #how often to step through scheduler

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            left_action = left_agent.choose_action(observation)
            right_action = right_agent.choose_action(observation)

            observation_, left_reward, right_reward, done = env.step(left_action, right_action)
            score+=left_reward + right_reward

            left_agent.store_transition(observation, left_action, left_reward, observation_, done)
            right_agent.store_transition(observation, right_action, right_reward, observation_, done)
            
            left_agent.learn()
            right_agent.learn()

            observation = observation_

        scores.append(score)
        eps_history.append(left_agent.epsilon)
        avg_score = np.mean(scores[-100:])

        if i % lr_step == 0:
            left_agent.Q_eval.scheduler.step()
            left_agent.Q_target.scheduler.step()
            right_agent.Q_eval.scheduler.step()
            right_agent.Q_target.scheduler.step()
            print(f'Learning Rate: {left_agent.Q_eval.scheduler.get_last_lr()}')

        if i % target_update == 0:
            left_agent.Q_target.load_state_dict(left_agent.Q_eval.state_dict())
            right_agent.Q_target.load_state_dict(right_agent.Q_eval.state_dict())

        print('episode',i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % left_agent.epsilon)
    
    x=[i+1 for i in range(n_games)]
    dir_name = './results_diagram/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    filename = dir_name+'Roadblock3.png'
    plotLearning(x, scores, eps_history, filename)
