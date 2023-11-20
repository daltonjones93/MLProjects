from torch import nn
import numpy as np



class nnModel(nn.Module):
    def __init__(self,input_size = 33,output_size = 1,n_hidden_units = 10, n_layers = 1, dropout_param = .05, activation = nn.Tanh()):
        super().__init__()
        self.layer1 = nn.Linear(input_size,n_hidden_units)
        self.middle_layers= [nn.Linear(n_hidden_units,n_hidden_units) for _ in range(n_layers)]

        self.last_layer = nn.Linear(n_hidden_units,output_size)

        self.dropout = nn.Dropout(dropout_param)
        self.activation = activation
    def forward(self,x):
        x = self.dropout(self.activation(self.layer1(x)))
        
        for layer in self.middle_layers:
            x = self.dropout(self.activation(layer(x)))
        return self.last_layer(x)