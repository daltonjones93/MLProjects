import torch

T = torch.tensor([[[.8,.1,.1],[.1,.6,.3]],[[.7,.2,.1],[.1,.8,.2]],[[.6,.2,.2],[.1,.4,.5]]])

R = torch.tensor([1,0,-1]).float()
gamma = .7
threshold = .0001
policy_optimal = torch.tensor([[1,0],[1,0],[1,0]]).float()


def policy_evaluation(policy, T, R, gamma, threshold):
    V = torch.zeros(R.shape[0])

    while True:
        V_temp = torch.zeros(R.shape[0])
        for state, actions in enumerate(policy):
            for action, action_prob in enumerate(actions):
                V_temp[state] += action_prob * (R[state] + gamma * torch.dot(T[state,action], V)) #take average over actions
        max_delta = torch.max(torch.abs(V-V_temp))
        V = V_temp.clone()
        if max_delta < threshold:
            break

    return V

print(policy_evaluation(policy_optimal, T, R, gamma, threshold))