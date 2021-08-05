import tensorly as tl
import neptune
from generate_data import RSE
import time
import numpy as np
import scipy

def als(factors, tensor, rank, rho, max_time, solve_method=None, method_steps=None, noise=None):
    factors=factors.copy()
    tensor_hat  = tl.cp_to_tensor((None, factors))  
    logging_val_old = RSE(tensor_hat, tensor)
    logging_val     = logging_val_old
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
                    factors[mode][i_column, :], _ = scipy.sparse.linalg.cg(X, rhs_column, x0 = factors[mode][i_column, :], maxiter=method_steps)
                    # print(f'💩 CG steps {_}')
            else:
                factors[mode] = (np.linalg.solve(inp.T @ inp + rho*eye, inp.T @ tar)).T
            mask[mode]=True
            t-=-1

            stop_time = time.time()
            tensor_hat  = tl.cp_to_tensor((None, factors))
            logging_time = stop_time - start_time
            logging_val_old = logging_val
            logging_val = RSE(tensor_hat, tensor)
            
            if noise is not None and logging_val < noise:
                return logging_time

            neptune.log_metric('RSE (i)', x=t, y=logging_val)
            neptune.log_metric('RSE (t)', x=logging_time, y=logging_val)  
            if logging_val_old < logging_val:
                print(f'Nazar 🐓 s♂️set c♂️ck: {logging_val - logging_val_old}')
                return logging_time
            if logging_time > max_time:
                return logging_time
            start_time += time.time() - stop_time



        

        
