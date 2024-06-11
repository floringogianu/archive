from turtle import forward
import torch 
from torch import nn
from collections import OrderedDict

class basic_mlp(nn.Module):
    def __init__(self,observation_shape,action_shape,hidden_size):
        super(basic_mlp,self).__init__()
        self.hidden_size = hidden_size
        self.model = nn.Sequential(OrderedDict([
            ('input_layer',nn.Linear(in_features=observation_shape,out_features=self.hidden_size)),
            ('relu_1',nn.ReLU()),
            ('hidden_layer_1',nn.Linear(in_features=self.hidden_size,out_features=self.hidden_size)),
            ('relu_2',nn.ReLU()),
            ('hidden_layer_2',nn.Linear(in_features=self.hidden_size,out_features=self.hidden_size)),
            ('relu_3',nn.ReLU()),
            ('output_layer',nn.Linear(in_features=self.hidden_size,out_features=action_shape))
        ]))

    def forward(self,observation):
        y = self.model(observation)
        return y 