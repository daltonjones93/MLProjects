import gym
import numpy as np
import torch

env = gym.make("CartPole-v1")#,render_mode = "human")



best_reward = 0
rewards = []
n_episodes = 10000
best_map = torch.rand(2,4)

for _ in range(n_episodes):
    env.reset()
    tmp = 0
    is_done = False
    tmp_map = torch.rand(2,4)


    while not is_done:
        action = torch.argmax(tmp_map @ torch.from_numpy(np.array(env.state)).float()).item()
        
        state,reward,terminated,truncated,info = env.step(action)
        is_done = terminated or truncated
        tmp += reward - np.log(abs(state[1])) - np.log(abs(state[3]))
    rewards.append(tmp)
    if tmp > best_reward:
        best_reward = tmp
        best_map = tmp_map
        

print(sum(rewards)/len(rewards))
print(rewards[:10])
print(best_reward)


env2 = gym.make("CartPole-v1",render_mode = "human")
env2.reset()
is_done = False
while not is_done:
    action = torch.argmax(best_map @ torch.from_numpy(np.array(env2.state)).float()).item()
    
    state,reward,terminated,truncated,info = env2.step(action)
    is_done = terminated or truncated
    # tmp += reward
    env2.render()


# env.reset()


# terminated = False
# truncated = False

# while not (terminated and truncated):
#     if env.state[1] > 0:
#         new_action = 0
#     else:
#         new_action = 1

#     new_state,reward,terminated, truncated,info = env.step(new_action)

#     env.render()