import torch
import numpy as np

T = torch.tensor([[[.8,.1,.1],[.1,.6,.3]],[[.7,.2,.1],[.1,.8,.2]],[[.6,.2,.2],[.1,.4,.5]]])

R = torch.tensor([1,0,-1]).float()
gamma = .7
action = 0
def calc_value(gamma,transition,rewards):
    inv = torch.eye(rewards.shape[0])-transition * gamma
    inv = torch.inverse(inv)
    value = torch.mm(inv,rewards.view(-1,1))
    return value

transition = T[:,action]
print(transition)
print(calc_value(gamma,transition,R))