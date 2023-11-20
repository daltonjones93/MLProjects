import torch
import gym

env = gym.make("FrozenLake-v1")#, render_mode = "human")
n_state = env.observation_space.n
print("Number of states: {}".format(n_state))
n_actions = env.action_space.n
print("Number of actions: {}".format(n_actions))

gamma = .99
threshold = .0001

def run_episode(env, policy):
    state = env.reset()[0]
    total_reward = 0
    is_done = False
    while not is_done:
        state,reward,terminated,truncated,_ = env.step(policy[state].item())
        is_done = terminated or truncated
        total_reward += reward
    return total_reward

def policy_evaluation(env,policy, gamma, threshold):
    n_state = env.observation_space.n
    n_action = env.action_space.n
    V = torch.zeros(policy.shape[0]) #value for each state
    
    while True:
        V_tmp = torch.zeros(policy.shape[0])
        for state in range(n_state):
            # v_actions = torch.zeros(n_actions)
            action = policy[state].item()

            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                V_tmp[state]+= trans_prob * (reward + gamma * V[new_state])
 
        max_delta = torch.max(torch.abs(V-V_tmp))
        V = V_tmp.clone()
        if max_delta < threshold:
            break
    
    return V

def policy_improvement(env, V, gamma):
    #basically compute argmax using V over actions
    n_state= env.observation_space.n
    n_actions = env.action_space.n
    
    policy = torch.zeros(V.shape[0])
    for state in range(n_state):
        v_actions = torch.zeros(n_actions) #need to find best action given V
        for action in range(n_actions):
            for trans_prob, new_state, reward, _ in env.env.P[state][action]:
                v_actions[action] += trans_prob*(reward + gamma * V[new_state])
        
        policy[state] = torch.argmax(v_actions)
    return policy

def policy_iteration(env, gamma, threshold):
    #loop back and forth until the policy stays the same.
    policy = torch.randint(high = env.action_space.n,size = (env.observation_space.n,))
    while True:
        V = policy_evaluation(env,policy, gamma, threshold)
        new_policy = policy_improvement(env,V,gamma)
        if torch.equal(policy,new_policy):
            break
        policy = new_policy
    return policy

opt_policy = policy_iteration(env,gamma,threshold)
print(opt_policy)







