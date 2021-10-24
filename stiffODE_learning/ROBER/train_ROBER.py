# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 05:11:11 2021

@author: LLSCAU
"""

# system package
import os, sys, argparse, logging
# basic dataprocess package
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None 
# customized package
import dataprocessor as dtp
from Robertson import RobertsonGenerator

def get_logger():
    # logger setting
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

if __name__ == '__main__':
    
    logger = get_logger()
    
    # parsing from shell params
    params = {}
    parser = argparse.ArgumentParser(description='Parameter Parser:')
    
    parser.add_argument('--device',         default='cpu',                               help='enable GPU or CPU')
    parser.add_argument('--sim_time',       default=(0.0,40.0), type=tuple,              help='simulation duration')
    parser.add_argument('--train_range',    default=(0.0,1.0), type=tuple,               help='initial condition range')
    parser.add_argument('--dt_pred',        default=0.1, type=float,                     help='predictive time step')
    parser.add_argument('--nb_ODE',         default=100, type=int,                       help='initial condition range')
    parser.add_argument('--pred_index',     default=['yA', 'yB', 'yC'], type=tuple,      help='expected sampling couple')
    parser.add_argument('--log_idx',        default=['yA', 'yB', 'yC'], type=tuple,      help='expected log transformed couple')    
    parser.add_argument('--feat_idx',       default=['yA', 'yB', 'yC'], type=tuple,      help='expected feature scaled couple')    
    parser.add_argument('--threshold',      default=1e-12, type=float,                   help='expected threshold to avoid zero values')
    parser.add_argument('--scaler_type',    default='StandardScaler', type=str,          help='expected feature scaler to datasets')
    parser.add_argument('--batch_size',     default=512, type=str,                       help='batch size for mini_batch training')
    
    args = parser.parse_args()
    
    #initializing processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))        
    # generate Robertson data
    ROBER = RobertsonGenerator(args.sim_time)
    data, ODE_initial_design = ROBER.generate_data(args.dt_pred, args.train_range, args.nb_ODE)
    # generate training/validation/test sets
    nb_ODE = data['sim_number'].max() + 1
    data_splitter = dtp.DataSplitter(nb_ODE=nb_ODE)
    id_ode_train, id_ode_val, id_ode_test = data_splitter.train_val_test(data)
    
    # predefined the data settings
    training_file = "../data/training_set.csv"
    val_file = "../data/val_set.csv"
    test_file = "../data/test_set.csv"
    
    pred_index = args.pred_index
    log_idx = args.log_index
    feat_idx= args.feat_idx
    
    composed = transforms.Compose([ dtp.AddThreshold(log_idx, args.threshold),
                                    dtp.LogTransformer(log_idx),
                                   ])    
    ODE_train = dtp.ODEDataset(training_file, pred_index, transform=composed, to_tensor=True)
    ODE_val = dtp.ODEDataset(val_file, pred_index, transform=composed, to_tensor=True)
    ODE_test = dtp.ODEDataset(test_file, pred_index, transform=composed, to_tensor=True)
    
    get_scaler = dtp.GetScaler(feat_idx, scaler_type=args.scaler_type)
    Xscaler, Yscaler = get_scaler(ODE_train)
    feature_scaling = dtp.FeatureScaling(feat_idx, (Xscaler, Yscaler))
    
    training_data = feature_scaling(ODE_train)
    val_data = feature_scaling(ODE_val)
    test_data = feature_scaling(ODE_test)

    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)