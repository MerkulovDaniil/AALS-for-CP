import tensorly as tl
import neptune
from generate_data import RSE
import time
import numpy as np
from oracles import *
import sys

def aam_min_scalar_iter(i, h, f_x, x, v, norm_prev, args):
    eye = np.eye(x.shape[-1])
    def check(h, args, forcereturn=False):
        f_loss, grad_f_loss, argmin_mode, tensor, rho, method_steps = args
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
            # x_new = y[:,None]
            f_x_new = []
            for j in range(grad_f_y.shape[0]):
                x_new[j][j] = (np.linalg.solve(X[j], Y[j])).T
                f_x_new.append(f_loss(x_new[j]))

            j_star = np.argmin(f_x_new)
            
            x_new = x_new[j_star]
            f_x_new=f_x_new[j_star]
            return True, ((y, f_y, grad_f_y , norm2_grad_f_y, x_new, f_x_new, j_star, h), forcereturn)
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

def aam_min_scalar(x, tensor, rank, rho, max_time, solve_method=None, method_steps=None):
    f_loss = lambda x : f(x, tensor, rho)
    grad_f_loss = lambda x : grad_f(x, tensor, rho)
    argmin_mode = lambda mode, x : argmin(mode, x, tensor, rho)
    args=f_loss, grad_f_loss, argmin_mode, tensor, rho, method_steps

    tensor_hat  = tl.cp_to_tensor((None, x))
    neptune.log_metric('RSE (i)', x=0, y=RSE(tensor_hat, tensor))
    neptune.log_metric('RSE (t)', x=0, y=RSE(tensor_hat, tensor))  
    
    mu=0 #ONLY!
    sa = 0.
    tau = 1.
    x = x.copy()
    v = x.copy()
    f_x = f_loss(x)
    h=np.ones(3, np.float64)

    i=0
    mode=0
    norm2_grad_f_y=None
    start_time = time.time()
    while True:
        # sys.stdout.write('\r'+f'🤖 AALS. Error {errors[-1]}')
        ret, forcereturn = aam_min_scalar_iter(i, h[mode], f_x, x, v, norm2_grad_f_y, args)
        print('\n')
        if forcereturn:
            print('restart\n')
            mu=0 #ONLY!
            sa = 0.
            tau = 1.
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
            v = x.copy()

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
        

        stop_time = time.time()
        tensor_hat  = tl.cp_to_tensor((None, x))
        logging_time = stop_time - start_time
        neptune.log_metric('RSE (i)', x=3*i, y=RSE(tensor_hat, tensor))
        neptune.log_metric('RSE (t)', x=logging_time, y=RSE(tensor_hat, tensor))  
    
        if logging_time > max_time:
            return logging_time
        start_time += time.time() - stop_time