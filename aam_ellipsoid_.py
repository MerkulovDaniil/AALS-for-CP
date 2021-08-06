import numpy as np
def ellipsoid(oracle, x0, P0, constraints, constraints_grad, f_min, tol=1e-6):
    '''
    Ellipsoid method from page 8 of 
    https://web.stanford.edu/class/ee364b/lectures/ellipsoid_method_notes.pdf 

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
            ``fun(x, *args) -> float``
        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
        
    grad : callable
        Method for computing the gradient vector of an objective 
        function. It should be a function that returns the gradient 
        vector:
            ``grad(x, *args) -> array_like, shape (n,)``
    
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    
    P0 : ndarray, shape (n,n)
        Initial guess of an ellipsoid. Array of real elements of size (n,n),
        where 'n' is the number of independent variables.

    constraints : List of callable, optional
    List of callable inequality constraints
        Each constraint is a callable function from x:
            ``constraint(x, *args) -> float``
        Constraints are taken in a following form:
            f_i(x) <= 0
    
    constraints_grad: List of callable, optional
        Method for computing the gradient vector of an corresponding constraint 
        function. Each element of a list should be a function that returns 
        the gradient vector:
            ``constraint_grad(x, *args) -> array_like, shape (n,)``

    '''

    x       = x0
    # f_prev  = fun(x0)
    xs      = []
    vols    = []
    P       = P0
    n       = len(x)
    f_best  = None
    while True:
        # x is infeasible. Constraint iteration
        alpha = None
        for constraint, constraint_grad in zip(constraints, constraints_grad):
            if constraint(x) > 0: 
                # print(f'😫 Violation of {constraint_grad(x)}')   
                g     = constraint_grad(x)
                # print(f'{g.T @ P @ g}')
                vol   = np.sqrt(g.T @ P @ g) 
                G     = 1/vol * g
                alpha = constraint(x)/vol
                # print(f'CONSTRAINT iteration vol {vol}') 
                break

        if alpha is None:
            # x is feasible. Objective iteration
            f, g   = oracle(x)
            # if f < f_min:
            #     return x
            vol = np.sqrt(g.T @ P @ g)
            G   = 1/vol * g

            # print(f'OBJECTIVE iteration vol {vol}') 

        Pg  = P@G
        if len(G.shape) < 2:
            G = np.expand_dims(G, axis=1)

        x   = x - 1/(n+1)*Pg
        # Monotonicity
        # if fun(x_new) > f_prev:
        #     return x, P
        # else:
        #     x = x_new

        P   = n**2/(n**2 - 1)*(P - 2/(n+1)* P @ G @ G.T @ P )
        # xs.append(x)
        # vols.append(vol)
        if vol <= tol:
            print(f'Я масенький {vol}, а вот ограничения:')
            print([constraint(x) for constraint in constraints])
            print(f'ХУЙ В РОТ {all([constraint(x) <= 0 for constraint in constraints]) == True}')


        if (vol <= tol) and all([constraint(x) <= 0 for constraint in constraints]) == True:
            return x0
        # print(f'New point {x}')

    # return x, P

def oracle_hull(w, xs):
    x = np.einsum('i,ijkl->jkl', w, np.array(xs))
    grad = grad_f_loss(x)
    g = np.array([(grad*(xs[0] - xs[2])).sum(), (grad*(xs[1] - xs[2])).sum()])
    return (f_loss(x), g)

def aam_ellipsoid_iter(i, h, f_x, x, v, norm_prev, args):
    eye = np.eye(x.shape[-1])
    def check(h, args, forcereturn=False):
        f_loss, grad_f_loss, argmin_mode, tensor, rho, sg_steps = args
        print(i,': ', h)
        y = v + h * (x-v)
        f_y = f_loss(y)
        grad_f_y = np.zeros_like(y)
        mask = np.ones(grad_f_y.shape[0],dtype=bool)
        X, Y = [], []
        for j in range(grad_f_y.shape[0]):    
            mask[j]=False
            inp = tl.tenalg.khatri_rao(y[mask])
            tar = tl.unfold(tensor, mode=j).T
            X.append(rho*eye + inp.T @ inp)
            Y.append(inp.T @ tar)
            grad_f_y[j] = (X[-1]@y[j].T - Y[-1]).T
            mask[j]=True

        if ((grad_f_y*(v-y)).sum() >= 0 and (f_x > f_y)) or forcereturn or i == 0:
            da, db, dc = grad_f_y
            da, db, dc = (da*da).sum(), (db*db).sum(), (dc*dc).sum()
            norm2_grad_f_y = da+db+dc
            x_new = [y.copy() for j in range(grad_f_y.shape[0])]
            f_x_new = []
            for j in range(grad_f_y.shape[0]):
                x_new[j][j] = (np.linalg.solve(X[j], Y[j])).T
                f_x_new.append(f_loss(x_new[j]))

            j_star = np.argmin(f_x_new)
           
            
            # x_new = x_new[j_star]
            f_x_new=f_x_new[j_star]
            mode=j_star
            g_con = []
            g_con.append(lambda x: np.array([-1,0],dtype=np.float64))
            g_con.append(lambda x: np.array([0,-1],dtype=np.float64))
            g_con.append(lambda x: np.array([1,0],dtype=np.float64))
            g_con.append(lambda x: np.array([0,1],dtype=np.float64))
            g_con.append(lambda x: np.array([1,1],dtype=np.float64))

            con=[]
            con.append(lambda x: -x[0])
            con.append(lambda x: -x[1])
            con.append(lambda x: x[0]-1)
            con.append(lambda x: x[1]-1)
            con.append(lambda x: sum(x)-1)


            w0 = np.zeros(2, dtype=np.float64)
            if j_star < 2:
                w0[j_star]=1

            def oracle_hull(w, xs):
                x = w[0]*xs[0] + w[1]*xs[1] + (1 - w[0] - w[1])*xs[2]
                # print(x.shape)
                # print(x_new[0].shape)
                grad = grad_f_loss(x)
                g = np.array([(grad*(xs[0] - xs[2])).sum(), (grad*(xs[1] - xs[2])).sum()])
                return (f_loss(x), g)
            
            # f = lambda w: oracle_hull(w, x_new)[0]
            # gf = lambda w: oracle_hull(w, x_new)[1]
            oracle = lambda w: oracle_hull(w, x_new)

            w= ellipsoid(oracle, w0, np.array([[1, 0], [0, 1]]), con, g_con, f_x_new)
            x_new = w[0]*x_new[0] + w[1]*x_new[1] + (1 - w[0] - w[1])*x_new[2]
            f_x_new=f_loss(x_new)

            return True, ((y, f_y, grad_f_y , norm2_grad_f_y, x_new, f_x_new, mode, h), forcereturn)
        else:
            return False, grad_f_y

    hl=None
    hr=np.float64(h)
    k=int(hr==1.0)
    fastreturn = True
    while True: #find right endpoint for line search
        is_ok, ret = check(hr, args)
        fastreturn = False
        if is_ok:
            return ret
        else:
            gr = ret
        if (gr*(x-v)).sum() < 0:
            hl=hr
            hr= 1 + 1e-8 * k**10
        else:
            break
        k-=-1
    
    #step left    
    tmp=max(0, min(hr-(1-(i+1)/(i+2)), (i+1)/(i+2))) #find left endpoint for line search
    k=1
    while hl==None:
        #print('-=2.5=-: [', hl, ', ', hr, ']')
        is_ok, ret = check(tmp, args)
        if is_ok:
            return ret
        else:
            gtmp = ret
        if (gtmp*(x-v)).sum() <= 0:
            hl=tmp
            break
        else:
            hr=tmp
        if tmp > 0:
             tmp = 1 - (1-tmp**(4)) #try this
        else:
            tmp = - 1e-8 * k**10
        k-=-1

    k=0
    while True:
        # print('[', hl, ', ', hr, ']')
        if hr < 0.8 or hl >= 1.:
            hc = hl + (hr-hl)*3/5
        else:
            hc = hl + (hr-hl)*4/5
        is_ok, ret = check(hc, args, hc==hr or hc==hl)
        k-=-1
        if is_ok:
            return ret
        else:
            gc = ret
        if (gc*(x-v)).sum() > 0:
            hr=hc
        else:
            hl=hc

def aam_ellipsoid(x, tensor, rank, rho, sg_steps, max_time):
    f_loss = lambda x : f(x, tensor, rho)
    grad_f_loss = lambda x : grad_f(x, tensor, rho)
    argmin_mode = lambda mode, x : argmin(mode, x, tensor, rho)
    args=f_loss, grad_f_loss, argmin_mode, tensor, rho, sg_steps

    tensor_hat  = tl.cp_to_tensor((None, x))
    neptune.log_metric('RSE (i)', x=0, y=RSE(tensor_hat, tensor))
    neptune.log_metric('RSE', y=RSE(tensor_hat, tensor), x=0)  
    
    mu=0 #ONLY!
    sa = 0.
    tau = 1.
    v = x.copy()
    f_x = f_loss(x)
    h=np.ones(3, np.float64)

    i=0
    mode=0
    norm2_grad_f_y=None
    start_time = time.time()
    while True:
        # sys.stdout.write('\r'+f'🤖 AALS. Error {errors[-1]}')
        ret, forcereturn = aam_ellipsoid_iter(i, h[mode], f_x, x, v, norm2_grad_f_y, args)

        if forcereturn:
            print('restart\n')
            mu=0 #ONLY!
            sa = 0.
            tau = 1.
            # x=warm(x)
            v = x.copy()
            if not restarted:
                restarted = True
                continue
            else:
                return logging_time
        restarted = False


        y, f_y, grad_f_y, norm2_grad_f_y, x, f_x, mode, t_tmp = ret
        h[mode]=t_tmp
        if f_x > f_y:
            return logging_time
            x=y.copy()
            f_x = f_y
            mu=0 #ONLY!
            sa = 0.
            tau = 1.
            v = (warm(x, rho)).copy()

        fxfy, vy = f_x-f_y, v-y
        ac = norm2_grad_f_y + 2*mu*fxfy
        bc = 2*mu*sa*fxfy + 2*tau*fxfy - mu*tau*((vy*vy).sum())
        cc = 2*sa*tau*fxfy
        a = (-bc + np.sqrt(bc*bc - 4*ac*cc)) / 2 / ac
        sa = sa + a
        v = tau*v + mu*a*y - a * (grad_f_y)
        tau = tau+mu*a
        v/=tau
        i-=-1
        print('a: ', a)
        print('\n')

        

        stop_time = time.time()
        tensor_hat  = tl.cp_to_tensor((None, x))
        logging_time = stop_time - start_time
        neptune.log_metric('RSE (i)', x=i, y=RSE(tensor_hat, tensor))
        neptune.log_metric('RSE', y=RSE(tensor_hat, tensor), x=logging_time)  
        if logging_time > max_time:
            return logging_time
        start_time += time.time() - stop_time

    