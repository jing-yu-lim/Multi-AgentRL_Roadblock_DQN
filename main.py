from agent import Agent
from utils import plotLearning
from environment import Roadblock_Env
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description = 'pomdpVSmdp')
parser.add_argument('-mdp', dest='observable', action='store_true')
parser.add_argument('-pomdp', dest='observable', action='store_false')
parser.set_defaults(observable = True)
parser.add_argument('-e', '--epsilon', nargs='?', type= float, default = 1.0, const = 1.0)
parser.add_argument('-g', '--gamenum', nargs='?', type =int, default = 3000, const = 3000)
parser.add_argument('-lr', '--lr', nargs='?', type =float, default = 0.003, const = 0.003)
parser.add_argument('-s', '--seed_value', nargs='?', type =int, default = 999, const = 999)
parser.add_argument('-lt', '--left_type', nargs='?', type =int, default = 0, const = 0)
parser.add_argument('-rt', '--right_type', nargs='?', type =int, default = 1, const = 1)

args= parser.parse_args()

if __name__ == '__main__':
    print(args)
    seed_value = args.seed_value #999
    np.random.seed(seed_value)
    if args.observable:
        input_dims = [4]
    else:
        input_dims = [2]

    env = Roadblock_Env(left_type=args.left_type, right_type = args.right_type, observable=args.observable) 
    left_agent = Agent(gamma = 0.99, epsilon = args.epsilon, batch_size = 64, n_actions=2, eps_min=0.01, input_dims=input_dims,lr = args.lr) 
    right_agent = Agent(gamma = 0.99, epsilon = args.epsilon, batch_size = 64, n_actions=2, eps_min=0.01, input_dims= input_dims,lr = args.lr)
    
    left_scores, right_scores, total_scores, eps_history = [], [], [], []
    
    n_games =args.gamenum
    target_update=50 
    lr_step=50       

    opt_game_count=0
    game_count =0
    for i in range(n_games):
        left_score,right_score, total_score, = 0, 0, 0
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

        if total_score >= 0.9:
            opt_game_count +=1

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
    
    type_name = 'Type_' + f'{args.left_type, args.right_type}_'
    if args.observable==True:
        dir_name = './results_diagram/' + 'MDP_eps' + f'{args.epsilon}_' + type_name[:-1] + '/'
    else:
        dir_name = './results_diagram/' + 'POMDP_eps' + f'{args.epsilon}_' + type_name[:-1] + '/'
        
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    if args.observable == True:
        leftpic= type_name + 'MDP_Left.png'
        filename = dir_name + leftpic
        plotLearning(x, left_scores, eps_history, filename, leftpic)
        rightpic=type_name + 'MDP_Right.png'
        filename = dir_name + rightpic
        plotLearning(x, right_scores, eps_history, filename, rightpic)
        totalpic=type_name +'MDP_Total.png'
        filename = dir_name + totalpic
        plotLearning(x, total_scores, eps_history, filename, totalpic)
    else:
        leftpic=type_name +'POMDP_Left.png'
        filename = dir_name + leftpic
        plotLearning(x, left_scores, eps_history, filename, leftpic)
        rightpic=type_name +'POMDP_Right.png'  
        filename = dir_name + rightpic
        plotLearning(x, right_scores, eps_history, filename, rightpic)
        totalpic=type_name +'POMDP_Total.png'
        filename = dir_name + totalpic
        plotLearning(x, total_scores, eps_history, filename, totalpic)
    
    print(args)
            
