import numpy as np
import random
import os
import wandb
from importlib import reload


import experimental_setup
reload(experimental_setup)
from experimental_setup import *

def cp_tensor_from_matrices(matrices, run_parameters):
    rank    = run_parameters['RANK']
    A, B, C = matrices
    I,J,K   = A.shape[0], B.shape[0], C.shape[0]
    T       = np.zeros((I, J, K))

    for p in range(rank):
        T += np.einsum('i,j,k->ijk', A[:, p], B[:, p], C[:, p])

    return T

def RSE(T_hat, T):
    return np.linalg.norm(T_hat - T)/np.linalg.norm(T)

# Reproducibility
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def init_wandb_log(run_parameters):
    WANDB_NAME          = run_parameters['WANDB_NAME']
    WANDB_TEAM          = run_parameters['WANDB_TEAM']
    WANDB_GROUP         = run_parameters['WANDB_GROUP']
    N_EXPERIMENTS       = run_parameters['N_EXPERIMENTS']
    DIM                 = run_parameters['DIM']
    RANK                = run_parameters['RANK']
    MODE                = run_parameters['MODE']
    REGULARIZATION_COEF = run_parameters['REGULARIZATION_COEF']
    NOISE               = run_parameters['NOISE']
    N_ITER              = run_parameters['N_ITER']
    LIST_OF_METHODS     = run_parameters['LIST_OF_METHODS']
    SEEDS               = run_parameters['SEEDS']
    method              = run_parameters['METHOD'] 
    i_seed              = run_parameters['SEED'] 
    i_exp               = run_parameters['I_EXP'] 

    wandb.init(entity = WANDB_TEAM, project=WANDB_NAME, name=method, group=WANDB_GROUP, reinit=True, notes = f'{i_exp+1}/{N_EXPERIMENTS}')
    wandb.config.n_iters     = N_ITER
    wandb.config.seed        = i_seed
    wandb.config.mode        = MODE
    wandb.config.tensor_dim  = DIM
    wandb.config.tensor_rank = RANK
    wandb.config.noise       = NOISE
    wandb.config.reg_coef    = REGULARIZATION_COEF
    wandb.config.n_runs      = N_EXPERIMENTS
    wandb.config.seed        = i_exp

def df_a(matrices, tensor, run_parameters):
    rho = run_parameters['REGULARIZATION_COEF']
    A, B, C = matrices
    return A@gamma_rho(B, C, rho) - np.einsum('ijk,jr,kr->ir', tensor, B, C)

def df_b(matrices, tensor, run_parameters):
    rho = run_parameters['REGULARIZATION_COEF']
    A, B, C = matrices
    return B@gamma_rho(C, A, rho) - np.einsum('ijk,ir,kr->jr', tensor, A, C)

def df_c(matrices, tensor, run_parameters):
    rho = run_parameters['REGULARIZATION_COEF']
    A, B, C = matrices
    return C@gamma_rho(A, B, rho) - np.einsum('ijk,ir,jr->kr', tensor, A, B)

def gamma_rho(A, B, run_parameters):
    rho = run_parameters['REGULARIZATION_COEF']
    return (A.T@A)*(B.T@B) + rho*np.eye(A.shape[1])

def df(matrices, tensor, run_parameters):
    return np.array( [df_a(matrices, tensor, run_parameters), 
                      df_b(matrices, tensor, run_parameters), 
                      df_c(matrices, tensor, run_parameters)])

def norm_of_the_gradient(matrices, tensor, run_parameters):
    da, db, dc = df(matrices, tensor, run_parameters)
    return np.sqrt((da*da).sum() + (db*db).sum() + (dc*dc).sum())

def f(matrices, tensor, run_parameters):
    return 0.5*((cp_tensor_from_matrices(matrices, run_parameters) - tensor)**2).sum()

    

        