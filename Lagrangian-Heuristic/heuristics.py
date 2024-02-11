import numpy as np
from numba import njit, objmode

import warnings

from numba.core.errors import NumbaDeprecationWarning, \
    NumbaPendingDeprecationWarning, NumbaPerformanceWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

from collections import namedtuple
from time import time

@njit(cache=True)
def compute_relative_gap(cost1, cost2, which="both", one=True):
    if cost1 == np.inf or cost2 == -np.inf:
        return 1.
    if cost1 == -np.inf or cost2 == np.inf:
        return -1.
    if which == "both":
        benchmark = max(abs(cost1),abs(cost2))
    elif which == "first":
        benchmark = abs(cost1)
    elif which == "second":
        benchmark = abs(cost2)
    if one:
        benchmark = max(benchmark,1)
    return (cost1-cost2)/benchmark


@njit(cache=True)
def trivial_soln(X,y):
    n,p = X.shape
    return np.zeros(p), np.copy(y)

@njit(cache=True)
def nb_set2arr(support):
    return np.array(list(support))

def set2arr(support):
    if len(support) == 0:
        return np.array([],dtype=int)
    else:
        return nb_set2arr(support)

@njit(cache=True)
def L0L2reg(beta,l0,l2,M):
    if np.abs(beta) > M:
        return np.inf
    return l0*(beta!=0) + l2*beta**2


@njit(cache=True)
def prox_L0L2reg(beta,l0,l2,M):
    val = np.abs(beta)/(1+2*l2)
    if val <= M:
        return np.sign(beta)*val if val>np.sqrt(2*l0/(1+2*l2)) else 0
    else:
        return np.sign(beta)*M if val>M/2 + l0/M/(1+2*l2) else 0


@njit(cache=True)
def quad_L0L2reg(x,a,b,l0,l2,M):
    return a*x**2+b*x+L0L2reg(x,l0,l2,M)


@njit(cache=True)
def Q_L0L2reg(a,b,l0,l2,M):
    beta = -b / (2*a)
    l0 /= (2*a)
    l2 /= (2*a)
    return prox_L0L2reg(beta,l0,l2,M)


@njit(cache=True)
def prox_L0L2reg_vec(beta,l0,l2,M):
    val = np.abs(beta)/(1+2*l2)
    thres1 = np.sqrt(2*l0/(1+2*l2))
    thres2 = M/2 + l0/M/(1+2*l2)
    return np.sign(beta)*np.where(val<=M, np.where(val>thres1, val, 0), np.where(val>thres2, M, 0))

@njit(cache=True)
def Q_L0L2reg_obj(a,b,l0,l2,M):
    beta = -b/(2*a)
    l0 = l0/(2*a)
    l2 = l2/(2*a)
    x = prox_L0L2reg(beta,l0,l2,M)
    return x, a*x**2+b*x+l0*(x!=0)+l2*x**2

@njit(cache=True)
def Q_L0L2reg_obj_vec(a,b,l0,l2,M):
    beta = -b / (2*a)
    l0_v = l0 / (2*a)
    l2_v = l2 / (2*a)
    x = prox_L0L2reg_vec(beta,l0_v,l2_v,M)
    return x, a*x**2+b*x+l0*(x!=0)+l2*x**2



@njit(cache=True)
def get_L0L2_cost(X, y, beta, r, l0, l2, M, active_set):
    loss = 0.5*r@r
    if len(active_set) == 0:
        return loss
    cost = loss + l0*np.sum(beta[active_set]!=0) + l2*np.sum(beta[active_set]**2)
    return cost


@njit(cache=True)
def L0L2_CD_loop(X, y, beta, l0, l2, M, S_diag, active_set, r):
    beta_old = 0.
    for i in active_set:
        beta_old = beta[i]
        Li = S_diag[i]
        gradi = -X[:,i]@r
        beta_tilde = beta_old - gradi/Li
        beta[i] = Q_L0L2reg(Li/2, -Li*beta_tilde, l0, l2, M)
        r = r - (beta[i] - beta_old)*X[:,i]
    return beta, r


@njit(cache=True)
def L0L2_CD(X, y, beta, cost, l0, l2, M, S_diag, active_set, r, rel_tol=1e-8, maxiter=3000, verbose=False):
    tol = 1
    old_cost = cost
    curiter = 0
    while tol > rel_tol and curiter < maxiter:
        old_cost = cost
        beta, r = L0L2_CD_loop(X, y, beta, l0, l2, M, S_diag, active_set, r)
        cost = get_L0L2_cost(X, y, beta, r, l0, l2, M, active_set)
        if verbose:
            print(cost)
        tol = abs(compute_relative_gap(old_cost, cost))
        curiter += 1
    return beta, cost, r



    
    
EPSILON = np.finfo('float').eps

def _initial_active_set(X, y, beta, upper_active_set_mask):
    p = X.shape[1]
    corr = np.corrcoef(X,y[:,None],rowvar=False)[:-1,-1]
    argpart = np.argpartition(-np.abs(corr), int(0.2*p))[:int(0.2*p)]
    active_set = set(argpart)
    
    active_set = active_set | set(np.where(np.abs(beta)>EPSILON*1e10)[0])
    
    active_set = active_set - set(np.where(~upper_active_set_mask)[0])
    
    active_set = np.array(sorted(active_set),dtype=int)
    return active_set


@njit(cache=True)
def _refined_initial_active_set(X, y, beta, l0, l2, M, S_diag, active_set, support, r):
    support.clear()
    num_of_similar_supports = 0
    delta_supp = 0
    while num_of_similar_supports < 3:
        delta_supp = 0
        beta, r = L0L2_CD_loop(X, y, beta, l0, l2, M, S_diag, active_set, r)
        for i in active_set:
            if (beta[i]!=0) and (i not in support):
                support.add(i)
                delta_supp += 1
        if delta_supp == 0:
            num_of_similar_supports += 1
        else:
            num_of_similar_supports = 0
    return support, beta, r


def _initialize_active_set_algo(X, y, l0, l2, M, S_diag, upper_active_set_mask, warm_start):
    p = X.shape[1]
    if S_diag is None:
        S_diag = np.linalg.norm(X, axis=0)**2
    if warm_start is not None:
        support, beta = warm_start['support'], np.copy(warm_start['beta'])
        beta[~upper_active_set_mask] = 0
        support = support - set(np.where(~upper_active_set_mask)[0])
        r = warm_start.get('r', y-X@beta)
        active_set = set2arr(support)
    else:
        beta, r = trivial_soln(X,y)
        active_set = _initial_active_set(X, y, beta, upper_active_set_mask)
        support = {0}
        support, beta, r = _refined_initial_active_set(X, y, beta, l0, l2, M, S_diag, active_set, support, r)
        
    return beta, r, support, S_diag


@njit(cache=True, parallel=True)
def _above_threshold_indices(l0,l2,M,X,y,S_diag,beta,r,upper_active_set_mask):
    grad = -r@X
    a = S_diag/2
    b = -a*2*beta+grad
    criterion = np.where(np.abs(b)/2/(a+l2)<=M, 4*l0*(a+l2)-b**2, a*M**2-np.abs(b)*M+l0+l2*M**2)
    above_threshold = np.where((criterion<0)&(upper_active_set_mask))[0]
    return above_threshold


@njit(cache=True)
def L0L2_ASCD(X, y, l0, l2, M, S_diag, beta, r, active_set, upper_active_set_mask, \
                cd_tol=1e-4, cd_max_itr=100, rel_tol=1e-6, kkt_max_itr=100, timing=False, maxtime=np.inf, verbose=False):
    start_time = 0.
    end_time = 0.
    if timing:
        with objmode(start_time='f8'):
            start_time = time()
    support = set(active_set)
    cost = get_L0L2_cost(X, y, beta, r, l0, l2, M, active_set)
    old_cost = cost
    if verbose:
        print("cost", cost)
    curiter = 0
    while curiter < kkt_max_itr:
        if timing:
            with objmode(end_time='f8'):
                end_time = time()
            if end_time - start_time > maxtime:
                break
        beta, cost, r = L0L2_CD(X, y, beta, cost, l0, l2, M, S_diag, active_set, r, cd_tol, cd_max_itr, verbose)
        if verbose:
            print("iter", curiter+1)
            print("cost", cost)
        above_threshold = _above_threshold_indices(l0,l2,M,X,y,S_diag,beta,r,upper_active_set_mask)
        outliers = list(set(above_threshold) - support)
        if len(outliers) == 0:
            if verbose:
                print("no outliers, computing relative accuracy...")
            if compute_relative_gap(cost, old_cost) < rel_tol or cd_tol < 1e-8:
                break
            else:
                cd_tol /= 100
                old_cost = cost
        
        support = support | set(outliers)
        if len(support) == 0:
            active_set = np.full(0,0)
        else:
            active_set = nb_set2arr(support)
        curiter += 1
    if curiter == kkt_max_itr:
        print('Maximum KKT check iterations reached, increase kkt_max_itr '
              'to avoid this warning')
    if timing and end_time - start_time > maxtime:
        print('Time limit reached, increase maxtime '
              'to avoid this warning')
    active_set = np.where(beta)[0]
    return beta, cost, r, active_set

def L0L2_ASCD_solve(X, y, l0, l2, M, upper_active_set_mask=None, S_diag=None, warm_start=None, \
                    cd_tol=1e-4, cd_max_itr=100, rel_tol=1e-6, kkt_max_itr=100, timing=False, maxtime=np.inf, verbose=False):
    
    n,p = X.shape
    if upper_active_set_mask is None:
        upper_active_set_mask = np.full(p,True)
    
    beta, r, support, S_diag = _initialize_active_set_algo(X, y, l0, l2, M, S_diag, upper_active_set_mask, warm_start)
    active_set = set2arr(support)
    
    beta, cost, r, active_set =  L0L2_ASCD(X, y, l0, l2, M, S_diag, beta, r, active_set, upper_active_set_mask, \
                                            cd_tol, cd_max_itr, rel_tol, kkt_max_itr, timing, maxtime, verbose)
    
    return beta, cost, r, set(active_set)


@njit(cache=True)
def L0L2_ASCDPSI(X, y, l0, l2, M, S_diag, beta, r, support, support_c, \
                 timing, maxtime, verbose):
    start_time = 0.
    end_time = 0.
    if timing:
        with objmode(start_time='f8'):
            start_time = time()
    flag = 0
    for i in support:
        for j in support_c:
            beta_bar = X[:, j] @ r + (X[:, i] @ X[:, j]) * beta[i]
            v = Q_L0L2reg(S_diag[j] / 2, -beta_bar, l0, l2, M)
            if np.abs(v) * np.sqrt(S_diag[j] + 2 * l2) > np.abs(beta[i]) * np.sqrt(S_diag[i] + 2 * l2):
                flag = 1
                cost_old = 0.5 * r @ r + l0 * np.sum(beta != 0) + l2 * np.sum(beta ** 2)
                beta[j] = v
                r = r + beta[i] * X[:, i] - beta[j] * X[:, j]
                beta[i] = 0
                cost_new = 0.5 * r @ r + l0 * np.sum(beta != 0) + l2 * np.sum(beta ** 2)
                print("Possible swap found, cost improve is ", cost_old - cost_new)
                break
        if flag == 1:
            break
        if timing:
            with objmode(end_time='f8'):
                end_time = time()
            if end_time - start_time > maxtime:
                break

    if timing and end_time - start_time > maxtime:
        print('Time limit reached, increase maxtime '
              'to avoid this warning')
    active_set = np.where(beta)[0]
    return beta, r, active_set, flag


def L0L2_ASCDPSI_solve(X, y, l0, l2, M, upper_active_set_mask=None, S_diag=None, warm_start=None, cd_tol=1e-4, \
                       cd_max_itr=100, rel_tol=1e-6, kkt_max_itr=100, PSI_max_itr=10, timing=False, \
                       maxtime=np.inf, verbose=False):
    n, p = X.shape
    if upper_active_set_mask is None:
        upper_active_set_mask = np.full(p, True)

    beta, r, support, S_diag = _initialize_active_set_algo(X, y, l0, l2, M, S_diag, upper_active_set_mask, warm_start)
    active_set = set2arr(support)

    PSI_itr = 0
    flag = 1
    while PSI_itr < PSI_max_itr and flag == 1:
        beta, cost, r, active_set = L0L2_ASCD(X, y, l0, l2, M, S_diag, beta, r, active_set, upper_active_set_mask, \
                                              cd_tol, cd_max_itr, rel_tol, kkt_max_itr, timing, maxtime, verbose)

        beta, r, active_set, flag = L0L2_ASCDPSI(X, y, l0, l2, M, S_diag, beta, r, active_set, \
                                                 set2arr(set(np.where(upper_active_set_mask)[0]) - set(active_set)),
                                                 timing, maxtime, verbose)

        PSI_itr += 1

    if PSI_itr == PSI_max_itr:
        print('Maximum Partial Swap Inescapable check iterations reached, increase PSI_max_itr '
              'to avoid this warning')

    return beta, cost, r, set(active_set)


def compute_next_lambda(X,y,l2,M,S_diag,beta,r,support):
    p = X.shape[1]
    if len(support) == p:
        return 0.
    grad = -r@X
    a = S_diag/2
    b = -a*2*beta+grad
    criterion = np.where(np.abs(b)/2/(a+l2)<=M, b**2/4/(a+l2), -a*M**2+np.abs(b)*M-l2*M**2)
    not_support = np.array(list(set(range(p))-support),dtype=int)
    return np.max(criterion[not_support])


def fit_path_L0L2(X, y, lambda2 = 0.01, M = np.inf, solver='ASCD',
                  lambda0_grid = None, maxSuppSize = None, n_lambda0 = 100, scaleDownFactor = 0.8, 
                  rel_tol=1e-6, cd_max_itr=100, kkt_max_itr=100, cd_tol=1e-4, verbose=True):
    assert solver in {"ASCD"}
    n,p = X.shape
    S_diag = np.linalg.norm(X, axis=0)**2

    if maxSuppSize is None:
        maxSuppSize = p
    _sol_str = 'beta sol_time support cost'
    Solution = namedtuple('Solution', _sol_str)
    sols = [] 
    terminate = False
    iteration_num = 0
    if verbose:
        print("L0L2 Heuristics Started.")
    if lambda0_grid is not None:
        lambda0_grid = sorted(lambda0_grid, reverse = True)
        beta = None
        while not terminate:
            l0 = lambda0_grid[iteration_num]
            st  = time()
            if verbose:
                print(l0,lambda2)
            if beta is not None:
                warm_start = {'beta':beta,'support':support}
            else:
                warm_start = None
            beta, cost, r, support = L0L2_ASCD_solve(X, y, l0, lambda2, M, upper_active_set_mask=None, S_diag=S_diag, warm_start=warm_start, \
                    cd_tol=cd_tol, cd_max_itr=cd_max_itr, rel_tol=rel_tol, kkt_max_itr=kkt_max_itr, timing=False, maxtime=np.inf, verbose=False)
            sols.append(Solution(beta=np.copy(beta), sol_time = time()-st, support = support, cost = cost))
            if verbose:
                print("Iteration: " + str(iteration_num) + ". Number of non-zeros: ",len(support))
            iteration_num += 1
            if iteration_num == len(lambda0_grid):
                terminate = True
            if len(support) >= maxSuppSize:
                terminate = True
                lambda0_grid = lambda0_grid[:iteration_num]
    else:
        beta, r = np.zeros(p), np.copy(y)
        support = set()
        lambda0_grid = []
        eps_factor = min(1e-3, (1-scaleDownFactor)*1e-2)
        while not terminate:
            if iteration_num == 0:
                l0 = compute_next_lambda(X,y,lambda2,M,S_diag,beta,r,support)*(1-eps_factor)
            else:
                l0 = min(l0*scaleDownFactor, compute_next_lambda(X,y,lambda2,M,S_diag,beta,r,support)*(1-eps_factor))
            lambda0_grid.append(l0)
            st  = time()
            if verbose:
                print(l0,lambda2)
            if beta is not None:
                warm_start = {'beta':beta,'support':support}
            else:
                warm_start = None
            beta, cost, r, support = L0L2_ASCD_solve(X, y, l0, lambda2, M, upper_active_set_mask=None, S_diag=S_diag, warm_start=warm_start, \
                    cd_tol=cd_tol, cd_max_itr=cd_max_itr, rel_tol=rel_tol, kkt_max_itr=kkt_max_itr, timing=False, maxtime=np.inf, verbose=False)
            sols.append(Solution(beta=np.copy(beta), sol_time = time()-st, support = support, cost = cost))
            if verbose:
                print("Iteration: " + str(iteration_num) + ". Number of non-zeros: ",len(support))
            iteration_num += 1
            if iteration_num == n_lambda0:
                terminate = True
            if len(support) >= maxSuppSize:
                terminate = True
            if l0 == 0:
                terminate = True
            
    return lambda0_grid, sols