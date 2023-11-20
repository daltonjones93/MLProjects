import gym
import numpy as np
import matplotlib.pyplot as plt
import torch


env = gym.make("CartPole-v1")
# env.reset()

def run_experiment(weight, env):
    env.reset()
    grads = []
    is_done = False
    total_rewards = 0
    while not is_done:
        state = torch.from_numpy(np.array(env.state)).float()
        probs = torch.nn.Softmax()(state @ weight)
        action = int(torch.bernoulli(probs[1]))
        d_softmax = torch.diag(probs)-probs.view(-1,1) * probs 
        d_log = d_softmax[action] / probs[action]
        grad = state.view(-1,1) * d_log
        state,reward,terminated,truncated,_ = env.step(action)
        total_rewards += reward
        is_done = terminated or truncated
        grads.append(grad)

    return total_rewards, grads

weight = torch.rand(4,2)

learning_rate = .001

n_episodes = 5000
rewards_over_time = []

for _ in range(n_episodes):
    tr, grads = run_experiment(weight, env)
    for i in range(len(grads)):
        weight += learning_rate * (tr-i) * grads[i]
    
    rewards_over_time.append(tr)

plt.figure()
plt.plot(rewards_over_time)
plt.show()







