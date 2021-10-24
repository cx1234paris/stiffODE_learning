# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 03:58:32 2021

@author: LLSCAU

This script is to generate robertson ode system dataset
"""

import scipy.integrate as integrate
import pandas as pd
import numpy as np
import os, sys
from pyDOE import lhs

# get the root path
root = os.path.join(os.getcwd(), "../..")


def ROBERode(t, y):
    k1 = 0.04
    k2 = 3e7
    k3 = 1e4
    yA, yB, yC = y
    # rhs 
    dyAdt = -k1*yA + k3*yB*yC
    dyBdt = k1*yA - k2*yB**2 - k3*yB*yC
    dyCdt = k2*yB**2
    return dyAdt, dyBdt, dyCdt

class RobertsonGenerator(object):
    
    def __init__(self, sim_time):
        """
        Args:.
            sim_time (tuple): starting and ending simulation time
        """ 
        self.sim_time = sim_time
        self.func_rhs = ROBERode

            
    def get_simulation(self, y0):
        solution = integrate.solve_ivp(self.func_rhs, self.sim_time, y0, method='BDF')
        t = solution.t
        y1, y2, y3 = solution.y
        return t, y1, y2, y3
    
    def generate_data(self, dt_pred, IC_range, nb_ODE, seed=444):
        """
        Args:
            dt_pred (float64): time interval of prediction(previous time and next time).
            IC_range (tuple): initial condition sampling range
            nb_ODE (float64): number of simulations to generate sampling datasets
        """ 
        u0_min = IC_range[0]
        u0_max = IC_range[1]
        column = ['yA0']
        np.random.seed(seed)
        ODE_initial_design = pd.DataFrame(data=lhs(1, samples=nb_ODE, criterion='maximin'), columns=column) #alpha_sampled
        ODE_initial_design['yA0'] = u0_min + (u0_max - u0_min)*ODE_initial_design['yA0']        
        # generate the final total data with each simulation time step points
        data = []
        for i, row in ODE_initial_design.iterrows():
            yA0 = [row['yA0'], 0.0, 1-row['yA0']]
            # previous step sequence data retrieving
            solution_X = integrate.solve_ivp(self.func_rhs, self.sim_time, yA0, method='BDF')
            times_X = solution_X.t
            times_Y = times_X + dt_pred
            solution_Y = integrate.solve_ivp(self.func_rhs, (self.sim_time[0],self.sim_time[1]+dt_pred),yA0,  
                                             method='BDF', t_eval=times_Y)
            # yA_X, yB_X, yC_X = solution_X.y
            # yA_Y, yB_Y, yC_Y = solution_Y.y
            cols_X = ['yA_X', 'yB_X', 'yC_X', 't']
            cols_Y = ['yA_Y', 'yB_Y', 'yC_Y', 't+dt']
            n_rows = len(times_X)
            n_cols_X = len(cols_X)
            n_cols_Y = len(cols_Y)
            arr_X = np.empty(shape=(n_rows, n_cols_X))
            arr_Y = np.empty(shape=(n_rows, n_cols_Y))
            for j, yx in enumerate(solution_X.y):
                arr_X[:, j] = yx 
            for j, yy in enumerate(solution_Y.y):
                arr_Y[:, j] = yy 
            arr_X[:, -1] = times_X
            arr_Y[:, -1] = times_Y
            df_X = pd.DataFrame(data=arr_X, columns=cols_X)
            df_Y = pd.DataFrame(data=arr_Y, columns=cols_Y)
            df = pd.concat([df_X, df_Y], axis=1).reset_index(drop=True)
            df['sim_number']=i
            data.append(df)
        data = pd.concat(data, axis=0).reset_index(drop=True)  
        if not os.path.exists(root + "/data/"):
            os.makedirs(root + "/data/")
        data.to_csv(root + "/data/ROBER.csv", sep=',')
        return data, ODE_initial_design


    
if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    
    # parser = argparse.ArgumentParser()
    ROBER = RobertsonGenerator((0.0, 10000.0))
    # data, ODE_initial_design = ROBER.generate_data(1.0, (0.0, 1.0), 300)

    t, y1, y2, y3 = ROBER.get_simulation([0.5, 0.0, 0.5])
    
    fig,axs = plt.subplots(3,1)
    axs[0].scatter(t, y1, c='r')
    axs[1].scatter(t, y2, c='b')
    axs[2].scatter(t, y3, c='g')
    plt.show()


