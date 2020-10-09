import numpy as np

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