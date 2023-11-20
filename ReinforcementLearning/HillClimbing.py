import gym
import numpy as np
import matplotlib.pyplot as plt
import torch

env = gym.make("CartPole-v1")#, render_mode = "human")
env.reset()

rewards_over_time = []

map = torch.rand(2,4)
best_score = 0
is_done = False
while not is_done:
    action = torch.argmax(map @ torch.from_numpy(np.array(env.state)).float()).item()
    state,reward,terminated,truncated,info = env.step(action)
    is_done = terminated or truncated
    best_score += reward - .05*np.abs(env.state[1])-.05*np.abs(env.state[3])


n_episodes = 1000

for _ in range(n_episodes):
    tmp_map = map + (.5)*(torch.rand(2,4)-.5)
    is_done = False
    new_score = 0
    env.reset()
    while not is_done:
        new_action = torch.argmax(tmp_map @ torch.from_numpy(np.array(env.state)).float()).item()
        state,reward,terminated,truncated,_ = env.step(new_action)
        new_score += reward - .05*np.abs(env.state[1])-.05*np.abs(env.state[3])
        is_done = terminated or truncated
    

    if new_score > best_score:
        map = tmp_map
        best_score = new_score
    rewards_over_time.append(best_score)

plt.figure()
plt.plot(rewards_over_time)
plt.show()

env2 = gym.make("CartPole-v1",render_mode = "human")
env2.reset()
is_done = False
while not is_done:
    new_action = torch.argmax(map @ torch.from_numpy(np.array(env2.state)).float()).item()
    state,reward,terminated,truncated,_ = env2.step(new_action)
    is_done = terminated or truncated





