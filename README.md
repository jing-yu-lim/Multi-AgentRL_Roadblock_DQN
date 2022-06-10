# Multi Agent RL Roadblock Environment
- to explore the differences between Fully Observed states (MDPs) and Partially Observed states (POMDPs)
- DQN implementation adapted from [Phil Atbor](https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/DeepQLearning)
# Environment Setup
- Scenario is such that two cars next two each other and are merging into a single lane
- Simple state space environment with two agents
- Single binary action space of either idling or advancing
## Reward
- Each agent will be randomly initiated to a Type, either 0 or 1:
  - type 0: if left car goes first, reward = 0.5 else reward = 0.4
  - type 1: if right car goes first, reward = 0.5 else reward = 0.4
  - type 2: receives reward = 0.4 if only one car advances
- If both agents advances, reward = -1
- If both agents diels, reward = -0.3
## POMDP vs MDP
- In the MDP environment, the Type of both agents will be observable to both agents
- In the POMDP environment, the Type will not be observable

# Results
- Blue curve: Epsilon 
- Orange scatter plot: Total Reward

| Agent Type | MDP | POMDP |
| -----------| --- | ----- |
| L=0 R=0 |![image](https://user-images.githubusercontent.com/79006977/173007875-92c9f0c2-a82f-436a-ae3a-e4c0b63b07bb.png) | ![image](https://user-images.githubusercontent.com/79006977/173007920-2aced974-56ea-4342-a033-28ad29a3d5dc.png) |
| L=0 R=2 | ![image](https://user-images.githubusercontent.com/79006977/173008346-8cae4d4f-c4ae-4639-ad26-fa1b3961db53.png) | ![image](https://user-images.githubusercontent.com/79006977/173008380-d6062e7a-cbb7-4d3f-8035-1b9ae6972d1e.png) |
| L=1 R=2 | ![image](https://user-images.githubusercontent.com/79006977/173008213-83a4671c-7f20-4bba-b8ef-92c6219019e4.png) | ![image](https://user-images.githubusercontent.com/79006977/173008256-0d16d54a-7f39-494c-8130-10b9ff93f303.png) |
| L=2 R=2 |![image](https://user-images.githubusercontent.com/79006977/173008967-af1f0e44-fef2-4173-a10b-fe17e919517d.png) | ![image](https://user-images.githubusercontent.com/79006977/173009017-7119e9c6-aa92-4183-8320-07a972953ad5.png) |

# Conclusion
- It appears that there is no difference between the POMDP and MDP setup. This could be because the agent easily overfit to the instantiated type over the course of the training

