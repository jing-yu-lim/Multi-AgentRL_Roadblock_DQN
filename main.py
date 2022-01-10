from agent import Agent
from utils import plotLearning
from environment import Roadblock_Env
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description = 'pomdpVSmdp')
parser.add_argument('-mdp', dest='observable', action='store_true')
parser.add_argument('-pomdp', dest='observable', action='store_false')
parser.add_argument('-e', '--epsilon', nargs='?', type= float, default = 1.0, const = 1.0)
parser.add_argument('-g', '--gamenum', nargs='?', type =int, default = 3000, const = 3000)
parser.add_argument('-lr', '--lr', nargs='?', type =float, default = 0.003, const = 0.003)
parser.add_argument('-s', '--seed_value', nargs='?', type =int, default = 999, const = 999)

args= parser.parse_args()

if __name__ == '__main__':
    print(args)

    seed_value = args.seed_value #999
    np.random.seed(seed_value)
    if args.observable:
        input_dims = [4]
    else:
        input_dims = [2]

    env = Roadblock_Env(left_type=0, right_type = 1, observable=args.observable) 
    left_agent = Agent(gamma = 0.99, epsilon = args.epsilon, batch_size = 64, n_actions=2, eps_min=0.01, input_dims=input_dims,lr = args.lr) 
    right_agent = Agent(gamma = 0.99, epsilon = args.epsilon, batch_size = 64, n_actions=2, eps_min=0.01, input_dims= input_dims,lr = args.lr)
    
    left_scores, right_scores, total_scores, eps_history = [], [], [], []
     
    n_games =args.gamenum
    target_update=50 #how often to update target network
    lr_step=50       #how often to step through scheduler

    for i in range(n_games):
        left_score,right_score, total_score = 0, 0, 0
        done = False
        observation = env.reset()
        # print(f'\noutside obs: {observation}')
        while not done:
            left_action = left_agent.choose_action(observation)
            right_action = right_agent.choose_action(observation)

            observation_, left_reward, right_reward, done = env.step(left_action, right_action)
            left_score +=left_reward
            right_score +=right_reward
            total_score+=left_reward + right_reward
            
            # print(f'\nleft action: {left_action} | rightaction: {right_action}')
            # print(f'left reward: {left_score} | right reward: {right_score}')
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

    if args.observable==True:
        dir_name = './results_diagram/' + 'MDP_eps' + f'{args.epsilon}' + '_rs' + f'{seed_value}' + '/'
    else:
        dir_name = './results_diagram/' + 'POMDP_eps' + f'{args.epsilon}' + '_rs' + f'{seed_value}' + '/'
        
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    if args.observable == True:
        filename = dir_name+'MDP_Roadblock_Left.png'
        plotLearning(x, left_scores, eps_history, filename)
        filename = dir_name+'MDP_Roadblock_Right.png'
        plotLearning(x, right_scores, eps_history, filename)
        filename = dir_name+'MDP_Roadblock_Total.png'
        plotLearning(x, total_scores, eps_history, filename)
    else:
        filename = dir_name+'POMDP_Roadblock_Left.png'
        plotLearning(x, left_scores, eps_history, filename)
        filename = dir_name+'POMDP_Roadblock_Right.png'
        plotLearning(x, right_scores, eps_history, filename)
        filename = dir_name+'POMDP_Roadblock_Total.png'
        plotLearning(x, total_scores, eps_history, filename)
    
    print(args)

    
