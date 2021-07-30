from tensorly.cp_tensor import validate_cp_rank
from tensorly.decomposition._cp import initialize_cp
from tensorly.cp_tensor import unfolding_dot_khatri_rao
from tensorly.cp_tensor import CPTensor
import numpy as np

def f(cp_tensor, tensor, rho):
    return 0.5*tl.norm(tl.cp_to_tensor(cp_tensor) - tensor)**2 + 0.5*rho * tl.norm(factors)**2

def grad_f(cp_tensor, tensor, rho):
    weights, factors = cp_tensor
    out = np.zeros_like(factors)
    # if orthogonalise:
    #         factors = [tl.qr(f)[0] if min(tl.shape(f)) >= rank else f for i, f in enumerate(factors)]
    modes_list = [mode for mode in range(tl.ndim(tensor))]
    Id = tl.eye(rank, **tl.context(tensor))*rho
    
    for mode in modes_list:
        
        pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
        for i, factor in enumerate(factors):
            if i != mode:
                pseudo_inverse = pseudo_inverse*tl.dot(tl.conj(tl.transpose(factor)), factor)
        pseudo_inverse += Id

        if weights is not None:
            # Take into account init weights
            mttkrp = unfolding_dot_khatri_rao(tensor, (weights, factors), mode)
        else:
            mttkrp = unfolding_dot_khatri_rao(tensor, (None, factors), mode)
        
        factor = factors[mode]
        out[mode] = tl.transpose(-tl.transpose(mttkrp) + tl.conj(pseudo_inverse)@tl.transpose(factor))
    return out

def argmin(mode, cp_tensor, tensor, rho):
    weights, factors = cp_tensor
    Id = tl.eye(rank, **tl.context(tensor))*rho
    pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
    for i, factor in enumerate(factors):
        if i != mode:
            pseudo_inverse = pseudo_inverse*tl.dot(tl.conj(tl.transpose(factor)), factor)
    pseudo_inverse += Id
    if weights is not None:
        # Take into account init weights
        mttkrp = unfolding_dot_khatri_rao(tensor, (weights, factors), mode)
    else:
        mttkrp = unfolding_dot_khatri_rao(tensor, (None, factors), mode)

    factor = tl.transpose(tl.solve(tl.conj(tl.transpose(pseudo_inverse)),
                            tl.transpose(mttkrp)))

    factors[mode] = factor
    # return cp_tensor