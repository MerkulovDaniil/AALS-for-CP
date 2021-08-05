import tensorly as tl
import numpy as np

def f(factors, tensor, rho):
    return 0.5*tl.norm(tl.cp_to_tensor((None, factors)) - tensor)**2 + 0.5*rho * tl.norm(factors)**2

def grad_f(factors, tensor, rho):
    # out = []
    out = np.zeros_like(factors)
    eye = np.eye(factors.shape[-1])
    mask = np.ones(factors.shape[0],dtype=bool)
    for mode in range(factors.shape[0]):    
        mask[mode]=False
        inp  = tl.tenalg.khatri_rao(factors[mask])
        tar = tl.unfold(tensor, mode=mode)
        
        # out.append (( - inp @ tar  +  (inp @ inp.T @ factors[mode].T)).T + rho*factors[mode])
        # out[mode] = ( - inp @ tar  +  ((inp.T @ inp +rho*eye)@ factors[mode].T)).T
        # out[mode] = ( - inp @ tar  +  inp @ inp.T@ factors[mode].T)).T + rho*factors[mode]
        # out[mode] =  -tar @ inp  +  factors[mode] @ (rho*eye + inp.T @ inp) #+ rho*factors[mode]
        out[mode] =  -tar @ inp  +  factors[mode] @ (rho*eye + inp.T @ inp) #+ rho*factors[mode]
        # out[mode] =  (-tar  +  factors[mode] @ inp.T) @ inp + rho*factors[mode]
        mask[mode]=True
    return out

def argmin(mode,factors,tensor,rho):
    mask = np.ones(factors.shape[0],dtype=bool)
    mask[mode]=False
    inp  = tl.tenalg.khatri_rao(factors[mask])
    tar = tl.unfold(tensor, mode=mode).T
    eye = np.eye(factors.shape[-1])
    factors[mode] = (np.linalg.solve(inp.T @ inp + rho*eye, inp.T @ tar)).T