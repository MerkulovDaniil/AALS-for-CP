import numpy as np
import random
import os
import wandb
from importlib import reload


import experimental_setup
reload(experimental_setup)
from experimental_setup import *

def cp_tensor_from_matrices(ABC, rank):
    A = ABC[0]
    B = ABC[1]
    C = ABC[2]
    I,J,K = A.shape[0], B.shape[0], C.shape[0]
    T = np.zeros((I, J, K))

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