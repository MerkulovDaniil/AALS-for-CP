import numpy as np
from sklearn import preprocessing
import random


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