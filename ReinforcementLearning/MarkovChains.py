import numpy as np
import torch
import matplotlib.pyplot as plt

#let's create a simple markov chain
T = torch.tensor([[.4,.6],[.8,.2]])
#recall, we're multiplying from the right for a mkv chain, always a strange convention.
T2 = torch.matrix_power(T, 2)
T5 = torch.matrix_power(T, 5)
T10 = torch.matrix_power(T, 10)
T15 = torch.matrix_power(T, 15)
T20 = torch.matrix_power(T, 20)

v = torch.tensor([[.7,.3]]) #row vector


print("Distribution of states after 1 step: \n{}".format(torch.mm(v,T)))

print("Distribution of states after 2 steps: \n{}".format(torch.mm(v,T2)))
print("Distribution of states after 5 steps: \n{}".format(torch.mm(v,T5)))
print("Distribution of states after 10 steps: \n{}".format(torch.mm(v,T10)))
print("Distribution of states after 15 steps: \n{}".format(torch.mm(v,T15)))
print("Distribution of states after 20 steps: \n{}".format(torch.mm(v,T20)))
