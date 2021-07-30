# import tensorly as tl

# init='svd'
# svd='numpy_svd'
# orthogonalise = False
# normalize_factors = True
# random_state=False


# weights, factors = initialize_cp(tensor, rank, init=init, svd=svd, 
#                                 random_state=random_state,
#                                 normalize_factors=normalize_factors)

######################################################################


# def scale(abc):
#     tensor_norm=1
#     for i in range(abc.shape[0]):
#         block_norm = tl.norm(abc[i])
#         abc[i] /= block_norm
#         tensor_norm *= block_norm
#     return abc * (tensor_norm**(1/abc.shape[0]))

# def warm(abc, rho):
#     mask = np.ones(abc.shape[0],dtype=bool)
#     for i in range(abc.shape[0]):
#         mask[i]=False
#         inp = tl.tenalg.khatri_rao(abc[mask])
#         tar = tl.unfold(tensor, mode=i).T
#         abc[i] = (np.linalg.solve(inp.T @ inp + rho, inp.T @ tar)).T
#         mask[i]=True
#     return abc


# def generate_starting_point(tensor, rank, rho):
#     a = preprocessing.normalize(np.random.random((tensor.shape[0], rank)), norm='l2')
#     b = preprocessing.normalize(np.random.random((tensor.shape[1], rank)), norm='l2')
#     c = preprocessing.normalize(np.random.random((tensor.shape[2], rank)), norm='l2')
#     abc = warm(np.array([a,b,c]), rho)
#     return scale(abc)