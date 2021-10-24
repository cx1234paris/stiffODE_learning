# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 01:09:19 2021

@author: LLSCAU
"""
# system package
import os, sys, argparse, logging
from logging.handlers import TimedRotatingFileHandler
# basic dataprocess package
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None 
# customized package for data
from lib import dataprocessor as dtp
from lib.Robertson import RobertsonGenerator
# customized package for model
from lib.ANN import ANN, rmse_error

###############################################################################
# import warnings filter: this must be corrected in the future but for now we don't care about it
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
###############################################################################
# get the root path
root = os.path.abspath(os.path.join(os.getcwd(), ".."))

def get_logger():
    # logger setting
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = TimedRotatingFileHandler(filename='test.log', when='M', interval=1, backupCount=3)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger    

if __name__ == '__main__':
    
    logger = get_logger()
###############################################################################
    
    #initializing processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    
    # generate Robertson data
    ROBER = RobertsonGenerator((0.0, 10000.0))
    data, ODE_initial_design = ROBER.generate_data(1.0, (0.0, 1.0), 300)
    # generate training/validation/test sets
    nb_ODE = data['sim_number'].max() + 1
    data_splitter = dtp.DataSplitter(nb_ODE=300)
    id_ode_train, id_ode_val, id_ode_test = data_splitter.train_val_test(data)
    
    # predefined the data settings
    training_file = root + "/data/training_set.csv"
    val_file = root + "/data/val_set.csv"
    test_file = root + "/data/test_set.csv"
    
    pred_index = ['yA', 'yB', 'yC']
    log_idx = ['yA', 'yB', 'yC']
    feat_idx= ['yA', 'yB', 'yC']
    
    composed = transforms.Compose([ dtp.AddThreshold(log_idx, 1e-12),
                                    # dtp.LogTransformer(log_idx),
                                   ])    
    ODE_train = dtp.ODEDataset(training_file, pred_index, transform=composed, to_tensor=True)
    ODE_val = dtp.ODEDataset(val_file, pred_index, transform=composed, to_tensor=True)
    ODE_test = dtp.ODEDataset(test_file, pred_index, transform=composed, to_tensor=True)
    
    get_scaler = dtp.GetScaler(feat_idx, scaler_type='PowerTransformer')
    Xscaler, Yscaler = get_scaler(ODE_train)
    feature_scaling = dtp.FeatureScaling(feat_idx, (Xscaler, Yscaler))
    
    training_data = feature_scaling(ODE_train)
    val_data = feature_scaling(ODE_val)
    test_data = feature_scaling(ODE_test)

###############################################################################    
    # initialize neural network model
    print(f'setup neural network model: type {ANN.__name__}')
    model = ANN(input_shape=len(pred_index), 
                output_shape=len(pred_index), 
                layers_form=(30, 30)
                ).to(device)
    print("Model structure: ", model, "\n\n")
    
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")    
        
    # Training parameters
    learning_rate = 5e-3
    batch_size = 512
    epochs = 30
    
    # Split the datasets for learning process
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    # Initialize the loss function and optimizer
    loss_fn = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), 
    #                           lr=learning_rate, 
    #                           betas=(0.9, 0.999), 
    #                           weight_decay=0.92,
    #                           amsgrad=False
    #                           )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, sample in enumerate(dataloader):
            X, y = sample['X(t)'].to(device), sample['X(t+dt)'].to(device)
            X, y = X.float(), y.float()
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        
            if batch % 3 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")    
                
    def validation(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, rmse = 0, 0
        with torch.no_grad():
            for sample in dataloader:
                X, y = sample['X(t)'].to(device), sample['X(t+dt)'].to(device)
                X, y = X.float(), y.float()
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                rmse += rmse_error(pred, y)
        test_loss /= num_batches
        rmse /= num_batches
        print(f"Validation Error: \n rmse: {(rmse):>0.1f},  Avg loss: {test_loss:>8f} \n")
    
    # run learning process
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        validation(val_dataloader, model, loss_fn)
    print("Done!")