from agent import Agent
from utils import plotLearning
from environment import Roadblock_Env
import numpy as np
import os
if __name__ == '__main__':
    env = Roadblock_Env()
    left_agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 64, n_actions=2, eps_min=0.01, input_dims=[4],lr = 0.003)
    right_agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 64, n_actions=2, eps_min=0.01, input_dims=[4],lr = 0.003)
    
    left_scores, right_scores, total_scores, eps_history = [], [], [], [] 
    n_games =1000
    target_update=50 #how often to update target network
    lr_step=50       #how often to step through scheduler

    for i in range(n_games):
        left_score,right_score, total_score = 0, 0, 0
        done = False
        observation = env.reset()
        #print(f'outside obs: {observation}')
        while not done:
            left_action = left_agent.choose_action(observation)
            right_action = right_agent.choose_action(observation)

            observation_, left_reward, right_reward, done = env.step(left_action, right_action)
            left_score +=left_reward
            right_score +=right_reward
            total_score+=left_reward + right_reward
            # print(f'\nleft action: {left_action} | rightaction: {right_action}')
            # print(f'left reward: {left_reward} | right reward: {right_reward}')
            # print(f"old obs: {observation} | new obs: {observation_} | done: {done}\n")


            left_agent.store_transition(np.copy(observation), left_action, left_reward, np.copy(observation_), done)
            right_agent.store_transition(np.copy(observation), right_action, right_reward, np.copy(observation_), done)
            
            left_agent.learn()
            right_agent.learn()

            observation = np.copy(observation_)

        left_scores.append(left_score)
        right_scores.append(right_score)
        total_scores.append(total_score)
        eps_history.append(left_agent.epsilon)
        avg_total_score = np.mean(total_scores[-100:])

        if i % lr_step == 0:
            left_agent.Q_eval.scheduler.step()
            left_agent.Q_target.scheduler.step()
            right_agent.Q_eval.scheduler.step()
            right_agent.Q_target.scheduler.step()
            print(f'Learning Rate: {left_agent.Q_eval.scheduler.get_last_lr()}')

        if i % target_update == 0:
            left_agent.Q_target.load_state_dict(left_agent.Q_eval.state_dict())
            right_agent.Q_target.load_state_dict(right_agent.Q_eval.state_dict())

        print('episode',i, 'left score %.2f' % left_score,'right score %.2f' % right_score, 'average score %.2f' % avg_total_score, 'epsilon %.2f' % left_agent.epsilon)
    
    x=[i+1 for i in range(n_games)]
    dir_name = './results_diagram/3000/'
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    filename = dir_name+'MPD_Roadblock_left.png'
    plotLearning(x, left_scores, eps_history, filename)
    filename = dir_name+'MPD_Roadblock_right.png'
    plotLearning(x, right_scores, eps_history, filename)
    filename = dir_name+'MPD_Roadblock_total.png'
    plotLearning(x, total_scores, eps_history, filename)
    
