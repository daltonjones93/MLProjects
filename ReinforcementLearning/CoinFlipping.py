import torch

gamma = 1 #do we need this to be less than 1 for a fixed point?
threshold = 1e-10
capital_max = 100
n_state = capital_max + 1 #(include 0)
head_prob = .4

rewards = torch.zeros(n_state)
rewards[-1] = 1.0

env = {'capital_max': capital_max,
       'head_prob':head_prob,
       'rewards':rewards,
       'n_state':n_state}

def value_iteration(env,gamma,threshold):
    head_prob = env['head_prob']
    n_state = env['n_state']
    capital_max = env['capital_max']
    V = torch.zeros(n_state)
    while True:
        V_temp = torch.zeros(n_state)
        for state in range(1,capital_max):
            v_actions = torch.zeros(n_state)
            for action in range(1,min(state,capital_max - state)+1):
                v_actions[action] += head_prob * (rewards[state + action] + gamma * V[state+action])
                v_actions[action] += (1 - head_prob) * (rewards[state - action] + gamma * V[state-action])
            
            V_temp[state] = torch.max(v_actions)
        max_delta = torch.max(torch.abs(V_temp-V))
        V = V_temp.clone()
        if max_delta < threshold:
            break
    return V

V_opt = value_iteration(env,gamma,threshold)

def compute_optimal_strategy(env,V_opt, gamma):
    n_state = env['n_state']
    head_prob = env['head_prob']
    capital_max = env['capital_max']

    policy = torch.zeros(capital_max).int()
    
    for state in range(1,capital_max):
        v_actions = torch.zeros(n_state)
        for action in range(1,min(state,capital_max-state)+1):
            v_actions[action] += head_prob * (rewards[state + action] + gamma * V_opt[state+action])
            v_actions[action] += (1 - head_prob) * (rewards[state - action] + gamma * V_opt[state-action])
        policy[state] = torch.argmax(v_actions)

    return policy

optimal_policy = compute_optimal_strategy(env,V_opt,gamma)

def optimal_strategy(capital):
    return optimal_policy[capital].item()
def random_strategy(capital):
    return torch.randint(1,capital+1,(1,)).item()
def conservative_strategy(capital):
    return 1

def run_episode(head_prob,capital,policy):
    while capital > 0:
        bet = policy(capital)
        if torch.rand(1).item() <head_prob:
            capital += bet
            if capital >= 100:
                return 1
        else:
            capital -= bet
    
    return 0

capital = 50
n_episode = 10000
n_win_random = 0
n_win_conservative = 0
n_win_optimal = 0
for episode in range(n_episode):
    n_win_random += run_episode(head_prob,capital,random_strategy)
    n_win_optimal += run_episode(head_prob,capital,optimal_strategy)
    n_win_conservative += run_episode(head_prob,capital,conservative_strategy)

print("Random win rate: {}".format(n_win_random/n_episode))
print("Conservative win rate: {}".format(n_win_conservative/n_episode))
print("Optimal win rate: {}".format(n_win_optimal/n_episode))


