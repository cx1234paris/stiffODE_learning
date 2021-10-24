# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 04:11:31 2021

@author: LLSCAU
"""
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import (FunctionTransformer, StandardScaler, MinMaxScaler, 
                                   RobustScaler, PowerTransformer, QuantileTransformer)
from sklearn.model_selection import train_test_split
pd.options.mode.chained_assignment = None  # default='warn'

# get the root path
root = os.path.join(os.getcwd(), "../..")


class DataSplitter(object):
    
    def __init__(self, nb_ODE, test_size=0.1, val_size=0.15):
        """
        Args:
            nb_ODE (int): number of simulation in dataset
            test_size(float, optional): test dataset proportion
            val_size(float, optional): validation dataset proportion(w.r.t the training set)
        """           
        self.id_ode = np.array([i for i in range(nb_ODE)])
        self.id_ode_train, self.id_ode_test = train_test_split(self.id_ode, 
                                                               test_size=test_size, random_state=24)
        self.id_ode_train, self.id_ode_val = train_test_split(self.id_ode_train, 
                                                              test_size=val_size, random_state=24)
    
    def train_val_test(self, data):
        df_ode_train = data[data['sim_number'].isin(self.id_ode_train)].reset_index(drop=True)
        df_ode_val = data[data['sim_number'].isin(self.id_ode_val)].reset_index(drop=True)
        df_ode_test = data[data['sim_number'].isin(self.id_ode_test)].reset_index(drop=True)
        df_ode_train.to_csv(root + "/data/training_set.csv", sep=',')
        df_ode_val.to_csv(root + "/data/val_set.csv", sep=',')
        df_ode_test.to_csv(root + "/data/test_set.csv", sep=',')
        print('dataset splitting finished')
        return self.id_ode_train, self.id_ode_val, self.id_ode_test


class ODEDataset(Dataset):
    """Robertson ODE system dataset."""
    
    def __init__(self, csv_file, pred_index, 
                 transform=None, to_tensor=False):
        """
        Args:
            csv_file (string): Path to the csv file with reference numerical solutions.
            pred_couple(list): input and output elements' index
            transform (callable, optional): Optional transform to be applied on input values.
            to_tensor (bool, optional): Optional transform to convert ndarrays in sample to Tensors.
        """        
        self.data = pd.read_csv(csv_file, sep=',')
        self.X_index = [str(X) + '_X' for X in pred_index]
        self.Y_index = [str(Y) + '_Y' for Y in pred_index]
        self.to_tensor = to_tensor
        if transform:
            self.data = transform(self.data)
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self, idx):
        X = self.data[self.X_index]
        Y = self.data[self.Y_index]
        sample = {'X(t)': X, 'X(t+dt)': Y}
        if self.to_tensor:
            X, Y = sample['X(t)'].to_numpy(), sample['X(t+dt)'].to_numpy()
            sample = {'X(t)': torch.from_numpy(X), 'X(t+dt)': torch.from_numpy(Y)}
            return {'X(t)': sample['X(t)'][idx], 'X(t+dt)': sample['X(t+dt)'][idx]}
        else:
            print('return a slice of pandas dataframe type instead of a tensor type')
            return {'X(t)': sample['X(t)'].iloc[idx, :], 'X(t+dt)': sample['X(t+dt)'].iloc[idx, :]}
            
        
class AddThreshold(object):
    """add threshold values into the sampling dataset to avoid 0 value for log case.
    Args:
        log_idx (list): Desired clipped terms. 
        threshold (float64): Desired threshold value
    """    
    def __init__(self, log_idx, threshold):
        self.X_index = [str(X) + '_X' for X in log_idx]
        self.Y_index = [str(Y) + '_Y' for Y in log_idx]
        self.threshold = threshold
        
    def __call__(self, data):
        data_thrX = data[self.X_index]
        data_thrY = data[self.Y_index]
        data_thrX[data_thrX <self.threshold] = self.threshold
        data_thrY[data_thrY <self.threshold] = self.threshold
        data[self.X_index] = data_thrX
        data[self.Y_index] = data_thrY
        return data        
    
        
class LogTransformer(object):
    """Log preprocess the target sampling dataset (specified by user's choice).
    Args:
        log_idx (list): Desired log transform terms. 
    """
    def __init__(self, log_idx):
        self.X_index = [str(X) + '_X' for X in log_idx]
        self.Y_index = [str(Y) + '_Y' for Y in log_idx]
        self.transformer = FunctionTransformer(np.log)
        
    def __call__(self, data):
        data[self.X_index] = self.transformer.transform(data[self.X_index])
        data[self.Y_index] = self.transformer.transform(data[self.Y_index])
        return data      


class GetScaler(object):
    """get the target scaler to feature scale the target sampling dataset.
    Args:
        feat_idx (list): Desired scaling transform terms.
        scaler_type(string): Desired scaler type
    """
    def __init__(self, feat_idx, scaler_type):
        self.X_index = [str(X) + '_X' for X in feat_idx]
        self.Y_index = [str(Y) + '_Y' for Y in feat_idx] 
        if scaler_type=="StandardScaler":
            self.scaler = StandardScaler()
        elif scaler_type=="MinMaxScaler":
            self.scaler = MinMaxScaler((-1, 1))
        elif scaler_type=="RobustScaler":
            self.scaler = RobustScaler()
        elif scaler_type=="PowerTransformer":
            self.scaler = PowerTransformer(method='box-cox')
        elif scaler_type=="QuantileTransformer":
            self.scaler = QuantileTransformer(output_distribution='normal')
        else:
            self.scaler = None   
            
    def __call__(self, Dataset):
        data_tr = Dataset.data.copy()
        X, Y = (data_tr[self.X_index], data_tr[self.Y_index])
        if self.scaler:
            Xscaler = self.scaler.fit(X)
            Yscaler = self.scaler.fit(Y)    
        return Xscaler, Yscaler
    

class FeatureScaling(object):
    """feature scale the target sampling dataset (specified by user's choice).
    Args:
        feat_idx (list): Desired scaling transform terms.
        scaler(tuple of sklearn scaler type): Desired scaler type
    """    
    def __init__(self, feat_idx, scaler):
        self.X_index = [str(X) + '_X' for X in feat_idx]
        self.Y_index = [str(Y) + '_Y' for Y in feat_idx] 
        self.Xscaler = scaler[0]
        self.Yscaler = scaler[1]
    
    def __call__(self, Dataset):
        data_tr = Dataset.data.copy()
        if self.Xscaler:
            trans_dtX = self.Xscaler.transform(data_tr[self.X_index])
            data_tr.loc[:, self.X_index] = trans_dtX
        if self.Yscaler:
            trans_dtY = self.Yscaler.transform(data_tr[self.Y_index])  
            data_tr.loc[:, self.Y_index] = trans_dtY  
        Dataset.data = data_tr
        return Dataset     

if __name__ == '__main__':    
    
    # generate training/validation/test sets
    data_file = root + "/data/ROBER.csv"
    data = pd.read_csv(data_file, sep=',')
    nb_ODE = data['sim_number'].max() + 1
    data_splitter = DataSplitter(nb_ODE)
    data_splitter.train_val_test(data)
    
    # predefined the data settings
    training_file = root + "/data/training_set.csv"
    val_file = root + "/data/val_set.csv"
    test_file = root + "/data/test_set.csv"
    
    pred_index = ['yA', 'yB', 'yC']
    log_idx = ['yA', 'yB', 'yC']
    feat_idx= ['yA', 'yB', 'yC']
    
    composed = transforms.Compose([ AddThreshold(log_idx, 1e-12),
                                    # LogTransformer(log_idx),
                                   ])    
    ODE_train = ODEDataset(training_file, pred_index, transform=composed, to_tensor=True)
    ODE_val = ODEDataset(val_file, pred_index, transform=composed, to_tensor=True)
    ODE_test = ODEDataset(test_file, pred_index, transform=composed, to_tensor=True)
    
    get_scaler = GetScaler(feat_idx, scaler_type='PowerTransformer')
    Xscaler, Yscaler = get_scaler(ODE_train)
    feature_scaling = FeatureScaling(feat_idx, (Xscaler, Yscaler))
    
    ODE_train_preprocessed = feature_scaling(ODE_train)
    ODE_val_preprocessed = feature_scaling(ODE_val)
    ODE_test_preprocessed = feature_scaling(ODE_test)
    
    ODE_train_preprocessed.data[['yA_X', 'yB_X', 'yC_X']].hist(bins=50)
