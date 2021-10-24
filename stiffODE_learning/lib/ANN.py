# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 01:17:57 2021

@author: LLSCAU
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Define model
class ANN(nn.Module):
    
    def __init__(self, input_shape, output_shape, layers_form, activation='relu'):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        self.linear_relu_stack = nn.Sequential()
        self.linear_relu_stack.add_module('input_layer', nn.Linear(input_shape, layers_form[0]))
        self.linear_relu_stack.add_module('input_layer_activation', self.activation)      
        
        for i in range(0, len(layers_form)):
            self.linear_relu_stack.add_module(f'hidden_layer_{i+1}', nn.Linear(layers_form[i], layers_form[i]))
            self.linear_relu_stack.add_module(f'hidden_layer_{i+1}_activation', self.activation) 
            
        self.linear_relu_stack.add_module('output_layer', nn.Linear(layers_form[-1], output_shape))
        self.linear_relu_stack.add_module('input_layer_activation', self.activation)    
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# customized error
def rmse_error(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

if __name__ == '__main__':       
    model = ANN(3, 3, (30, 30)).to(device)
    print("Model structure: ", model, "\n\n")
    
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")