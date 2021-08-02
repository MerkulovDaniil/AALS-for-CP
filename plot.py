import numpy as np
from scipy.spatial.distance import cdist
import math
%matplotlib inline
import matplotlib.pyplot as plt
import copy
from numpy import linalg
import time

def pltt(data, labels, colors, xlabel=None, ylabel=None, n_grid=100):
    n_algs = len(data)
    
    x_max = np.zeros(n_algs, np.float64)
    x_min = np.zeros(n_algs, np.float64)
    for k in range(n_algs):
        x_arrays, y_arrays = [], []
        
        if len(data[k])==1:
            y_arrays = data[k][0]
            for i in range(len(y_arrays)):
                x_arrays.append(np.arange(len(y_arrays[i])))
        elif len(data[k])==2:
            x_arrays = data[k][0]
            y_arrays = data[k][1]
            
        for i in range(len(x_arrays)):
            x_max[k] = min(x_arrays[i][-1], x_arrays[i-1][-1])
            x_min[k] = max(x_arrays[i][0], x_arrays[i-1][0])
    
    x_min=max(x_min)
    x_max=min(x_max)
    
    x_grid = np.linspace(x_min, x_max, n_grid)
    
    y_mean = np.zeros((n_algs, n_grid), np.float64)
    y_max = np.zeros((n_algs, n_grid), np.float64)
    y_min = np.zeros((n_algs, n_grid), np.float64)
    std_dev=np.zeros_like(y_mean)
    
    for k in range(n_algs):
        x_arrays, y_arrays = [], []
        
        if len(data[k])==1:
            y_arrays = data[k][0]
            for i in range(len(y_arrays)):
                x_arrays.append(np.arange(len(y_arrays[i])))
        elif len(data[k])==2:
            x_arrays = data[k][0]
            y_arrays = data[k][1]
        x_arrays, y_arrays = np.array(x_arrays), np.array(y_arrays)
        
        
        n=len(x_arrays) 
        
        y_interp = np.zeros((n, n_grid), np.float64)
        mask_array = []
        for i in range(n):
            mask = (x_arrays[i] >= x_min)*(x_arrays[i] <= x_max)
            mask_array.append(mask)
        
        for i in range(n):
            y_interp[i] = np.interp(x_grid, (x_arrays[i])[mask_array[i]], (y_arrays[i])[mask_array[i]])
            y_mean[k]+=y_interp[i]
        
        y_mean[k]/=n
        
        ### max-min var ###
        #y_max[k] = y_interp.max(axis=0)
        #y_min[k] = y_interp.min(axis=0)
        
        if n == 1:
            continue
            
        for i in range(n):
            std_dev[k] += (y_interp[i] - y_mean[k])**2
        std_dev[k] /= (n-1)
        std_dev[k] = np.sqrt(std_dev[k])
        #y_max[k] = y_mean[k] + std_dev[k]
        #y_min[k] = y_mean[k] - std_dev[k]
        #f_opt = min(y_min[k].min(), y_min[k-1].min())
    f_opt = y_mean.min()
    #y_mean-=f_opt

    ### std var ###
    y_max = y_mean + std_dev
    y_min = y_mean - std_dev
    
    fig, ax = plt.subplots()
    for k in range(n_algs):
        ax.semilogy(x_grid, y_mean[k], color= colors[k], label=labels[k])
        ax.fill_between(x_grid, y_min[k], y_max[k], color=colors[k], alpha=0.3, linewidth=0)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    leg = ax.legend();
    ax.grid(axis='both')

    plt.grid(True)
    # plt.savefig('/content/drive/My Drive/colab/ACC-SIN-std.png', dpi=200, bbox_extra_artists=(leg, ax), bbox_inches='tight')


def pltr(data, labels, colors, xlabel=None, ylabel=None, n_grid=100):

    data_new=[]
    x_max=[]
    y_max=[]

    for i, params in enumerate(params_list):
        data_new=[]
        for j, alg in enumerate(data[i]):
            data_new.append([[],[]])
            x = data[i][j][0]
            y = data[i][j][1]
            if j == 0:
                x_max.append(x[-1])
                y_max.append(y[0])
            else:
                x_max[-1] = min(x_max[-1], x[-1])
                y_max[0] = max(y_max[-1], x[-1])

    n_algs = len(data_new)
    for i, params in enumerate(params_list):
        for j, alg in enumerate(data[i]):
            mask = data[i][j][0] < x_max[i]
            mask[np.argmin(mask)] = True
            data[i][j][0] = data[i][j][0][mask] / x_max[i]
            data[i][j][1] = data[i][j][1][mask]

            n = len(data[i][j][1])
            data[i][j][1] *= np.linspace(1/data[i][j][1][0], 1, n)
            

            data_new[j][0].append(data[i][j][0])
            data_new[j][1].append(data[i][j][1])
           
    data =  data_new    
    
    x_grid = np.linspace(0, 1, n_grid)
    
    y_mean = np.zeros((n_algs, n_grid), np.float64)
    y_max = np.zeros((n_algs, n_grid), np.float64)
    y_min = np.zeros((n_algs, n_grid), np.float64)
    std_dev=np.zeros_like(y_mean)
    fig, ax = plt.subplots()
    for k in range(n_algs):
        x_arrays = data[k][0]
        y_arrays = data[k][1]
        # x_arrays, y_arrays = np.array(x_arrays), np.array(y_arrays)
        
        n=len(x_arrays) 
        
        y_interp = np.zeros((n, n_grid), np.float64)
        
        for i in range(n):
            # print(x_arrays[i])
            y_interp[i] = np.interp(x_grid, x_arrays[i], y_arrays[i])
            # ax.semilogy(x_grid, y_interp[i], color= colors[k], label=None, alpha=0.2)
            y_mean[k]+=y_interp[i]
        
        y_mean[k]/=n
        
        ### max-min var ###
        y_max[k] = y_interp.max(axis=0)
        y_min[k] = y_interp.min(axis=0)
        
        if n == 1:
            continue
            
        for i in range(n):
            std_dev[k] += (y_interp[i] - y_mean[k])**2
        std_dev[k] /= (n-1)
        std_dev[k] = np.sqrt(std_dev[k])
        #y_max[k] = y_mean[k] + std_dev[k]
        #y_min[k] = y_mean[k] - std_dev[k]
        #f_opt = min(y_min[k].min(), y_min[k-1].min())
    f_opt = y_mean.min()
    #y_mean-=f_opt

    ### std var ###
    # y_max = y_mean + std_dev
    # y_min = y_mean - std_dev
    
    mask=x_grid<0.9
    for k in range(n_algs):
        ax.semilogy(x_grid[mask], y_mean[k][mask], color= colors[k], label=labels[k], alpha=0.8)
        
        # ax.fill_between(x_grid, y_min[k], y_max[k], color=colors[k], alpha=0.3, linewidth=0)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    leg = ax.legend();
    ax.grid(axis='both')

    plt.grid(True)
    # plt.savefig('cp-av.png', dpi=200, bbox_extra_artists=(leg, ax), bbox_inches='tight')
