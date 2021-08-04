import tensorly as tl
import neptune
from generate_data import RSE
import time
import numpy as np
import scipy

def als(factors, tensor, rank, rho, max_time, solve_method=None, method_steps=None):
    factors=factors.copy()
    tensor_hat  = tl.cp_to_tensor((None, factors))  
    neptune.log_metric('RSE (i)', x=0, y=RSE(tensor_hat, tensor))
    neptune.log_metric('RSE (t)', x=0, y=RSE(tensor_hat, tensor))  
    
    t=0
    start_time = time.time()
    mask = np.ones(factors.shape[0],dtype=bool)
    eye = np.eye(factors.shape[-1])
    while True:
        for mode in range(factors.shape[0]):
            mask[mode]=False
            inp  = tl.tenalg.khatri_rao(factors[mask])
            tar = tl.unfold(tensor, mode=mode).T

            if solve_method == 'np.linalg.solve':
                factors[mode] = (np.linalg.solve(inp.T @ inp + rho*eye, inp.T @ tar)).T
            elif solve_method == 'cg':
                X = inp.T @ inp + rho*eye
                Y = inp.T @ tar
                for i_column, rhs_column in enumerate(Y.T):
                    factors[mode][i_column, :], _ = scipy.sparse.linalg.cg(X, rhs_column, maxiter=method_steps)
            else:
                factors[mode] = (np.linalg.solve(inp.T @ inp + rho*eye, inp.T @ tar)).T

            mask[mode]=True
        t-=-3

        if True and t % 30 == 0:
            stop_time = time.time()
            tensor_hat  = tl.cp_to_tensor((None, factors))
            logging_time = stop_time - start_time
            neptune.log_metric('RSE (i)', x=t, y=RSE(tensor_hat, tensor))
            neptune.log_metric('RSE (t)', x=logging_time, y=RSE(tensor_hat, tensor))  
            if logging_time > max_time:
                return logging_time
            start_time += time.time() - stop_time

