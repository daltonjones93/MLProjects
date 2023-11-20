import torch
import gym

env = gym.make("FrozenLake-v1")#, render_mode = "human")
n_state = env.observation_space.n
print("Number of states: {}".format(n_state))
n_actions = env.action_space.n
print("Number of actions: {}".format(n_actions))

gamma = .99
threshold = .0001

def value_iteration(env, gamma, threshold):
    n_state= env.observation_space.n
    n_action = env.action_space.n
    V = torch.zeros(n_state) #this is what we compute over time
    while True:
        V_temp = torch.empty(n_state)
        for state in range(n_state):
            v_actions = torch.zeros(n_action)
            for action in range(n_actions):
                for transition_prob, new_state, reward, _ in env.env.P[state][action]:
                    v_actions[action] += transition_prob * (reward + gamma * V[new_state]) #this takes expected value

            V_temp[state] = torch.max(v_actions)
        max_delta = torch.max(torch.abs(V-V_temp))
        V = V_temp.clone()
        if max_delta < threshold:
            break
    return V

def compute_optimal_policy(env,V_optimal,gamma):
    n_state= env.observation_space.n
    n_action = env.action_space.n
    optimal_policy = torch.zeros(n_state)
    for state in range(n_state):
        v_actions = torch.zeros(n_actions)
        for action in range(n_action):
            for trans_prob, new_state,reward,_ in env.env.P[state][action]:
                v_actions[action] += trans_prob * (reward + gamma*V_optimal[new_state])
        
        optimal_policy[state] = torch.argmax(v_actions)
    return optimal_policy

def run_episode(env, policy):
    state = env.reset()[0]
    total_reward = 0
    is_done = False
    while not is_done:
        state,reward,terminated,truncated,_ = env.step(policy[state].item())
        is_done = terminated or truncated
        total_reward += reward
    return total_reward


V_optimal = value_iteration(env,gamma,threshold)
print(V_optimal)
optimal_policy = compute_optimal_policy(env,V_optimal,gamma)
print(optimal_policy)

n_episodes = 1000
avg_reward = 0
for _ in range(n_episodes):
    avg_reward += run_episode(env,optimal_policy)/n_episodes
print(avg_reward)
