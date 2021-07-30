import numpy as np
from importlib import reload
import time
import sys
import tensorly
from scipy.optimize import minimize_scalar
#zalupa 
# External files
import utils
import experimental_setup
reload(utils)
reload(experimental_setup)
from utils              import *
from experimental_setup import *

def cp_als(matrices, tensor, run_parameters):
    rank        = run_parameters['RANK']
    n_iter      = run_parameters['N_ITER']
    max_time    = 60
    rho         = run_parameters['REGULARIZATION_COEF']
    i_exp       = run_parameters['I_EXP']
    A, B, C     = matrices

    errors = []
    wtime=[]
    tensor_hat  = cp_tensor_from_matrices(matrices, run_parameters)
    errors.append(RSE(tensor_hat, tensor)) 
    wtime.append(0)
    i = 0
    wandb.log({ 'Wall time': wtime[-1],
                'RSE': errors[-1],
                'Loss': f(matrices, tensor, run_parameters),
                'Iterations': i,
                'Norm of the gradient': norm_of_the_gradient(matrices, tensor, run_parameters)})
    
    
    start_time = time.perf_counter()
    for i in range(n_iter):
        sys.stdout.write('\r'+f'{i_exp}🤖 ALS. Error {errors[-1]}')
        # optimize a
        input_a  = tensorly.tenalg.khatri_rao([B, C])
        target_a = tensorly.unfold(tensor, mode=0).T
        A = (np.linalg.solve(input_a.T @ input_a, input_a.T @ target_a)).T

        # optimize b
        input_b  = tensorly.tenalg.khatri_rao([A, C])
        target_b = tensorly.unfold(tensor, mode=1).T
        B = (np.linalg.solve(input_b.T @ input_b, input_b.T @ target_b)).T

        # optimize c
        input_c  = tensorly.tenalg.khatri_rao([A, B])
        target_c = tensorly.unfold(tensor, mode=2).T
        C = (np.linalg.solve(input_c.T @ input_c, input_c.T @ target_c)).T
        
        stop_time   = time.perf_counter()
        matrices    = (A,B,C)
        tensor_hat  = cp_tensor_from_matrices(matrices, run_parameters)
        errors.append(RSE(tensor_hat, tensor))
        start_time += time.perf_counter() - stop_time

        wtime.append(time.perf_counter()-start_time)
        wandb.log({ 'Wall time': wtime[-1],
                    'RSE': errors[-1],
                    'Loss': f(matrices, tensor, run_parameters),
                    'Iterations': i,
                    'Norm of the gradient': norm_of_the_gradient(matrices, tensor, run_parameters)})
    wandb.config.rse_start   = errors[0]
    wandb.config.rse_finish = errors[-1]
    wandb.config.n_iters     = i
    return np.array(wtime), np.array(errors)

def cp_aals(matrices, tensor, run_parameters):
    rank        = run_parameters['RANK']
    n_iter      = run_parameters['N_ITER']
    max_time    = 60
    rho         = run_parameters['REGULARIZATION_COEF']
    i_exp       = run_parameters['I_EXP']
    A, B, C     = matrices
    matrices_v  = np.array([A,B,C])
    momentum_A  = 0 
    tau         = 1
    
    # Here matrices stands for the x_k, while
    # matrices_v stands for the v_k

    errors = []
    wtime=[]
    tensor_hat  = cp_tensor_from_matrices(matrices, run_parameters)
    errors.append(RSE(tensor_hat, tensor)) 
    wtime.append(0)
    i = 0
    wandb.log({ 'Wall time': wtime[-1],
                'RSE': errors[-1],
                'Loss': f(matrices, tensor, run_parameters),
                'Iterations': i,
                'Norm of the gradient': norm_of_the_gradient(matrices, tensor, run_parameters)})
    
    start_time = time.perf_counter()
    for i in range(3*n_iter):
        sys.stdout.write('\r'+f'{i_exp}🤖 AALS. Error {errors[-1]}')

        # Best convex combination
        beta_optimal = line_search_for_beta(matrices, matrices_v, tensor, run_parameters)
        matrices_y = matrices + beta_optimal*(matrices_v - matrices)

        # Alternating step
        gradients       = df(matrices_y, tensor, run_parameters)
        norms_of_grads  = [np.linalg.norm(df_x) for df_x in gradients]
        ind_max_grad    = np.argmax(norms_of_grads)
        A, B, C         = matrices
        if ind_max_grad == 0:
            # optimize a
            input_a  = tensorly.tenalg.khatri_rao([B, C])
            target_a = tensorly.unfold(tensor, mode=0).T
            A = (np.linalg.solve(input_a.T @ input_a, input_a.T @ target_a)).T
        elif ind_max_grad == 1:
            # optimize b
            input_b  = tensorly.tenalg.khatri_rao([A, C])
            target_b = tensorly.unfold(tensor, mode=1).T
            B = (np.linalg.solve(input_b.T @ input_b, input_b.T @ target_b)).T
        elif ind_max_grad == 2:
            # optimize c
            input_c  = tensorly.tenalg.khatri_rao([A, B])
            target_c = tensorly.unfold(tensor, mode=2).T
            C = (np.linalg.solve(input_c.T @ input_c, input_c.T @ target_c)).T
        else:
            raise Exception("Problem in alternating step")
        
        matrices = np.array([A,B,C])

        # Euclidian Proximal Step
        f_x = f(matrices,   tensor, run_parameters)
        f_y = f(matrices_y, tensor, run_parameters)
        sqared_grad_norm= np.array([np.linalg.norm(df_x)**2 for df_x in gradients]).sum()
        temp = (f_y - f_x)*2*tau/sqared_grad_norm
        momentum_a = (temp + np.sqrt(max(temp**2 + 4*temp*momentum_A, 0)))/2
        momentum_A = momentum_A + momentum_a
        matrices_v = matrices_v - momentum_a*gradients

        if i % 3 == 0:
            stop_time   = time.perf_counter()
            tensor_hat  = cp_tensor_from_matrices(matrices, run_parameters)
            errors.append(RSE(tensor_hat, tensor))
            start_time += time.perf_counter() - stop_time

            wtime.append(time.perf_counter()-start_time)
            wandb.log({ 'Wall time': wtime[-1],
                        'RSE': errors[-1],
                        'Loss': f(matrices, tensor, run_parameters),
                        'Iterations': i//3,
                        'Norm of the gradient': norm_of_the_gradient(matrices, tensor, run_parameters),
                        'beta': beta_optimal})
    wandb.config.rse_start   = errors[0]
    wandb.config.rse_finish = errors[-1]
    wandb.config.n_iters     = i//3
    return np.array(wtime), np.array(errors)

def line_search_for_beta(matrices, matrices_v, tensor, run_parameters):
    def f_scalar(beta):
        return f(matrices + beta*(matrices_v - matrices),  tensor, run_parameters)
    
    res = minimize_scalar(f_scalar, bounds = [0,1])
    return res.x



def acc_cp_als(abc, tensor, run_parameters):
    rank        = run_parameters['RANK']
    n_iter      = run_parameters['N_ITER']
    max_time    = 60
    rho         = run_parameters['REGULARIZATION_COEF']
    i_exp       = run_parameters['I_EXP']
    A,B,C = abc
    
    f_loss = lambda abc: 0.5*((cp_tensor_from_matrices(abc,rank) - tensor)**2).sum()/np.linalg.norm(tensor)**2
    def grad_f_loss(abc, tensor, rho=rho):
        a,b,c = abc
        def gamma_rho(a,b, rho=rho):
            return (a.T@a)*(b.T@b) + rho*np.eye(a.shape[1])         

        def g_a(a,b,c,tensor, rho=rho):
            return a@gamma_rho(b,c, rho) - np.einsum('ijk,jr,kr->ir', tensor,b,c)
            
        def g_b(a,b,c,tensor, rho=rho):
            return b@gamma_rho(c,a, rho) - np.einsum('ijk,ir,kr->jr', tensor,a,c)

        def g_c(a,b,c,tensor, rho=rho):
            return c@gamma_rho(a,b, rho) - np.einsum('ijk,ir,jr->kr', tensor,a,b)
        return 1/np.linalg.norm(tensor)**2 * np.array([g_a(*abc, tensor), g_b(*abc, tensor), g_c(*abc, tensor)])

    mu=0 #ONLY!
    AA = 0.
    tau = 1.
    x = np.array([A,B,C])
    v = x.copy()
    
    wtime=[0]
    errors = []
    tensor_hat  = cp_tensor_from_matrices(x,rank)
    errors.append(RSE(tensor_hat, tensor))
    i=0
    wandb.log({'RSE': errors[-1],
               'Iterations': int(i/3),
               'Wall time': wtime[-1]})
    start_time = time.perf_counter()
    f_x = None
    args=f_loss, grad_f_loss, tensor
    #args=A,B,C
    t=np.ones(3, np.float64)
    for i in range(int(3*n_iter)):
        sys.stdout.write('\r'+f'{i_exp}🤖 AALS. Error {errors[-1]}')
        y, f_y, grad_f_y, norm2_grad_f_y, x, f_x, t[i%3] = agmsdr_iter(i, t[i%3], f_x, x, v, args)
        #nplot
        # y = v + t * (x-v)
        #print('\n')
        g = (f_x - f_y)
        a = norm2_grad_f_y + 2*mu*g
        b = 2*mu*AA*g + 2*tau*g - mu*tau*(((v - y)*(v - y)).sum())
        c = 2*AA*tau*g

        alpha = (-b + np.sqrt(max(b*b - 4*a*c, 0))) / 2 / a
        
        AA = AA + alpha

        
        v = tau*v + mu*alpha*y - alpha * (grad_f_y)
        tau = tau+mu*alpha
        v/=tau
        stop_time = time.perf_counter()
        tensor_hat  = cp_tensor_from_matrices(x,rank)
        errors.append(RSE(tensor_hat, tensor))
        start_time += time.perf_counter() - stop_time

        wtime.append(time.perf_counter()-start_time)
        if i%3 == 0:
            wandb.log({'RSE': errors[-1],
                       'Iterations': int(i/3 + 1),
                       'Wall time': wtime[-1],
                       'Norm of the gradient': norm2_grad_f_y})
    wandb.config.rse_start   = errors[0]
    wandb.config.rse_finish = errors[-1]
    wandb.config.n_iters     = int(i/3 + 1)
    return np.array(wtime), np.array(errors)
    
            
def agmsdr_iter(i, t, f_x, x, v, args):
    f_loss, _, _ = args
    def check(t, args, forcereturn=False):
        f_loss, grad_f_loss, tensor = args
        #print(i,': ', t)
        y = v + t * (x-v)
        f_y = f_loss(y)
        grad_f_y = grad_f_loss(y, tensor)

        if ((grad_f_y*(v-y)).sum() >= 0 and f_x >= f_y ) or forcereturn:
            
            da, db, dc = grad_f_y
            da, db, dc = (da*da).sum(), (db*db).sum(), (dc*dc).sum()
            norm2_grad_f_y = da+db+dc
            
            x_new = y.copy()    
            A,B,C=y
            
            I = np.argmax([da, db, dc])
            if I == 0:
                input_a  = tensorly.tenalg.khatri_rao([B, C])
                target_a = tensorly.unfold(tensor, mode=0).T
                A = (np.linalg.solve(input_a.T @ input_a, input_a.T @ target_a)).T
            if I == 1:
                input_b  = tensorly.tenalg.khatri_rao([A, C])
                target_b = tensorly.unfold(tensor, mode=1).T
                B = (np.linalg.solve(input_b.T @ input_b, input_b.T @ target_b)).T
            if I == 2:
                input_c  = tensorly.tenalg.khatri_rao([A, B])
                target_c = tensorly.unfold(tensor, mode=2).T
                C = (np.linalg.solve(input_c.T @ input_c, input_c.T @ target_c)).T
            x_new = np.array([A,B,C])
            f_x_new=f_loss(x_new)
            return True, (y, f_y, grad_f_y , norm2_grad_f_y, x_new, f_x_new, t) #f(x_new) can be optimized
        else:
            return False, grad_f_y

    if f_x==None: f_x = f_loss(x) 
    tl=None
    tr=np.float32(t)
    k=0
    while True: #find right endpoint for line search
        is_ok, ret = check(tr, args)
        if is_ok:
            return ret
        else:
            gr = ret
        if (gr*(x-v)).sum() < 0:
            tl=tr
            tr= 1 + 1e-8 * k**10
        else:
            break
        k-=-1

    #print('-=1=-: [', tl, ', ', tr, ']')
    
    #step left    
    tmp=max(0, min(tr-(1 - (i+1)/(i+2)), (i+1)/(i+2))) #find left endpoint for line search
    k=1
    while tl==None:
        #print('-=2.5=-: [', tl, ', ', tr, ']')
        is_ok, ret = check(tmp, args)
        if is_ok:
            return ret
        else:
            gtmp = ret
        if (gtmp*(x-v)).sum() <= 0:
            tl=tmp
            break
        else:
            tr=tmp
        
        tmp = 1 - (1-tmp**(4)) #try this
        k-=-1

    #print('-=2=-: [', tl, ', ', tr, ']')

    
    while True:
        # print('[', tl, ', ', tr, ']')
        if tr < 0.8 or tl >= 1.:
            tc = tl + (tr-tl)*3/5
        else:
            tc = tl + (tr-tl)*4/5
        alarm = (tc==tr or tc==tl)
        is_ok, ret = check(tc, args, tc==tr or tc==tl)

        if is_ok:
            return ret
        else:
            gc = ret
        if (gc*(x-v)).sum() > 0:
            tr=tc
        else:
            tl=tc