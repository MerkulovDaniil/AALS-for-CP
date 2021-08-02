import numpy as np
from sklearn import preprocessing
import random
import tensorly as tl

def generate_3d_tensor(sizes, rank, mu):
    if type(sizes) == int:
        I,J,K = sizes, sizes, sizes
    else:
        I,J,K = sizes

    a = preprocessing.normalize(np.random.randn(I, rank), norm='l2')
    b = preprocessing.normalize(np.random.randn(J, rank), norm='l2')
    c = preprocessing.normalize(np.random.randn(K, rank), norm='l2')
    N = np.random.randn(I,J,K)
    
    tensor = cp_tensor_from_matrices((a,b,c))
    tensor += mu*np.linalg.norm(tensor)/np.linalg.norm(N)*N

    return tensor
    # return tensor, a,b,c

def cp_tensor_from_matrices(factors):
    a,b,c = factors
    return np.einsum('ip,jp,kp->ijk', a, b, c)

def cp_tensor_from_matrices(factors):
    return tl.cp_to_tensor((None, factors))

def RSE(tensor_hat, tensor):
    return np.linalg.norm(tensor_hat - tensor)/np.linalg.norm(tensor)



def scale(factors):
    tensor_norm=1
    for mode in range(factors.shape[0]):
        block_norm = tl.norm(factors[mode])
        factors[mode] /= block_norm
        tensor_norm *= block_norm
    return factors * (tensor_norm**(1/factors.shape[0]))

def warm(factors, rho):
    mask = np.ones(factors.shape[0],dtype=bool)
    eye = np.eye(factors.shape[-1])
    for mode in range(factors.shape[0]):
        mask[mode]=False
        inp  = tl.tenalg.khatri_rao(factors[mask])
        tar = tl.unfold(tensor, mode=mode).T
        factors[mode] = (np.linalg.solve(inp.T @ inp + rho*eye, inp.T @ tar)).T
        mask[mode]=True
    return factors


def generate_starting_point(tensor, rank, rho):
    a = preprocessing.normalize(np.random.random((tensor.shape[0], rank)), norm='l2')
    b = preprocessing.normalize(np.random.random((tensor.shape[1], rank)), norm='l2')
    c = preprocessing.normalize(np.random.random((tensor.shape[2], rank)), norm='l2')
    factors = warm(np.array([a,b,c]), rho)
    return scale(factors)