def generate_3d_tensor(sizes = 30, rank = 20, mu = 1e-7, mode = 'random'):
    if type(sizes) == int:
        I,J,K = sizes, sizes, sizes
    else:
        I,J,K = sizes
    if mode == 'random':
        A = preprocessing.normalize(np.random.randn(I, rank), norm='l2')
        B = preprocessing.normalize(np.random.randn(J, rank), norm='l2')
        C = preprocessing.normalize(np.random.randn(K, rank), norm='l2')
        N = np.random.randn(I,J,K)
        
        T = cp_tensor_from_matrices((A,B,C),rank)
        
        T += mu*np.linalg.norm(T)/np.linalg.norm(N)*N
    elif mode == 'ill':
        A_l = preprocessing.normalize(np.random.randn(I, rank), norm='l2')
        B_l = preprocessing.normalize(np.random.randn(J, rank), norm='l2')
        C_l = preprocessing.normalize(np.random.randn(K, rank), norm='l2')

        A_r = preprocessing.normalize(np.random.randn(rank, rank), norm='l2')
        B_r = preprocessing.normalize(np.random.randn(rank, rank), norm='l2')
        C_r = preprocessing.normalize(np.random.randn(rank, rank), norm='l2')
        
        bad_vec = np.zeros((rank,))
        left_len = int(rank/2)
        right_len = len(bad_vec) - left_len
        bad_vec[:left_len] = np.random.uniform(int(I*J*K/2), I*J*K, left_len)
        bad_vec[-right_len:] = np.random.uniform(0, 1, right_len)
        A_d = np.diag(bad_vec)

        bad_vec = np.zeros((rank,))
        left_len = int(rank/2)
        right_len = len(bad_vec) - left_len
        bad_vec[:left_len] = np.random.uniform(int(I*J*K/2), I*J*K, left_len)
        bad_vec[-right_len:] = np.random.uniform(0, 1, right_len)
        B_d = np.diag(bad_vec)

        bad_vec = np.zeros((rank,))
        left_len = int(rank/2)
        right_len = len(bad_vec) - left_len
        bad_vec[:left_len] = np.random.uniform(int(I*J*K/2), I*J*K, left_len)
        bad_vec[-right_len:] = np.random.uniform(0, 1, right_len)
        C_d = np.diag(bad_vec)
        
        A = A_l@A_d@A_r
        B = B_l@B_d@B_r
        C = C_l@C_d@C_r

        N = np.random.randn(I,J,K)
        
        T = cp_tensor_from_matrices((A,B,C),rank)
        
        T += mu*np.linalg.norm(T)/np.linalg.norm(N)*N
    elif mode == 'ill_spreaded':
        A_l = preprocessing.normalize(np.random.randn(I, rank), norm='l2')
        B_l = preprocessing.normalize(np.random.randn(J, rank), norm='l2')
        C_l = preprocessing.normalize(np.random.randn(K, rank), norm='l2')

        A_r = preprocessing.normalize(np.random.randn(rank, rank), norm='l2')
        B_r = preprocessing.normalize(np.random.randn(rank, rank), norm='l2')
        C_r = preprocessing.normalize(np.random.randn(rank, rank), norm='l2')
        
        bad_vec = np.zeros((rank,))
        left_len = int(rank/2)
        right_len = len(bad_vec) - left_len
        bad_vec[:left_len] = np.random.uniform(int(I*J*K/2), I*J*K, left_len)
        bad_vec[-right_len:] = np.random.uniform(0, 1, right_len)
        A_d = np.diag(bad_vec)

        bad_vec = np.zeros((rank,))
        left_len = int(rank/2)
        right_len = len(bad_vec) - left_len
        bad_vec[:left_len] = np.random.uniform(int(I*J*K/2), I*J*K, left_len)
        bad_vec[-right_len:] = np.random.uniform(0, 1, right_len)
        B_d = np.diag(bad_vec)

        bad_vec = np.zeros((rank,))
        left_len = int(rank/2)
        right_len = len(bad_vec) - left_len
        bad_vec[:left_len] = np.random.uniform(int(I*J*K/2), I*J*K, left_len)
        bad_vec[-right_len:] = np.random.uniform(0, 1, right_len)
        C_d = np.diag(bad_vec)
        
        A = A_l@A_d@A_r
        B = B_l@B_d@B_r
        C = C_l@C_d@C_r

        N = np.random.randn(I,J,K)
        
        T = cp_tensor_from_matrices((A,B,C),rank)
        
        T += mu*np.linalg.norm(T)/np.linalg.norm(N)*N
    return T