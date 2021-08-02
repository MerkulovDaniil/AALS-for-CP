import tensorly as tl
import neptune
from generate_data import RSE
import time
import numpy as np

def am(factors, tensor, rank, rho, max_time=60, verbose=True):
    tensor_hat  = tl.cp_to_tensor((None, factors))  
    neptune.log_metric('RSE (i)', x=0, y=RSE(tensor_hat, tensor))
    neptune.log_metric('RSE', y=RSE(tensor_hat, tensor), x=0)  
    
    t=0
    start_time = time.time()
    mask = np.ones(factors.shape[0],dtype=bool)
    eye = np.eye(factors.shape[-1])
    while True:
        for mode in range(factors.shape[0]):
            mask[mode]=False
            inp  = tl.tenalg.khatri_rao(factors[mask])
            tar = tl.unfold(tensor, mode=mode).T
            factors[mode] = (np.linalg.solve(inp.T @ inp + rho*eye, inp.T @ tar)).T
            mask[mode]=True
        t-=-3

        if verbose and t % 30 == 0:
            stop_time = time.time()
            tensor_hat  = tl.cp_to_tensor((None, factors))
            logging_time = stop_time - start_time
            neptune.log_metric('RSE (i)', x=t, y=RSE(tensor_hat, tensor))
            neptune.log_metric('RSE', y=RSE(tensor_hat, tensor), x=logging_time)  
            if logging_time > max_time:
                return logging_time
            start_time += time.time() - stop_time