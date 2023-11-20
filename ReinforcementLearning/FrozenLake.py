import gym
import torch
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("FrozenLake-v1")#, render_mode = "human")
n_state = env.observation_space.n
print("Number of states: {}".format(n_state))
n_actions = env.action_space.n
print("Number of actions: {}".format(n_actions))


def run_episode(env, policy):
    state = env.reset()[0]
    total_reward = 0
    is_done = False
    while not is_done:
        state,reward,terminated,truncated,_ = env.step(policy[state].item())
        is_done = terminated or truncated
        total_reward += reward
    return total_reward

env.reset()

random_policy = torch.randint(high = n_actions, size = (n_state,))
best_reward = run_episode(env,random_policy)

best_policy = torch.randint(high = n_actions,size = (n_state,))

n_episodes = 1000
avg_reward = 0


for _ in range(n_episodes):
    tmp_reward = 0
    tmp_policy = torch.randint(high = n_actions, size = (n_state,))
    for _ in range(10):
        tmp_reward += run_episode(env,tmp_policy)/10.0
    if tmp_reward > best_reward:
        best_policy = tmp_policy
        best_reward = tmp_reward

avg_reward = 0
for _ in range(n_episodes):
    avg_reward += run_episode(env,best_policy)/n_episodes
print(avg_reward)


# is_done = False
# while not is_done:
#     state,reward,terminated,truncated,_ = env.step(env.action_space.sample())
#     is_done = terminated or truncated

