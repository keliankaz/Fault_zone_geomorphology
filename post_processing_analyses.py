#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:07:13 2020

@author: kdascher
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

## a series of functions and scripts for model post-processing
model_name = 'arctan1.00e+00_exp2.00e-02_fault04-03-2020/arctan1.00e+00_exp2.00e-02_fault.pkl'

def load_model(file_name):
    with open(file_name, 'rb') as f:
        rmg, ds = pickle.load(f)
    return rmg, ds

def model_variance(rmg):
    # GRID should be of dimensions nx,ny,nt
    time_std = rmg.topographic__elevation.std(dim='time')
    time_stdy = time_std.mean(dim='y')
    plt.figure()
    plt.plot(time_stdy,np.arange(time_stdy.shape[0]))
    plt.show()
    plt.xlabel('average Standard devation by row')
    plt.ylabel('row')

rmg, ds = load_model(model_name)
model_variance(ds)


