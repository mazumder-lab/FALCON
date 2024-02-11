from errno import ENETUNREACH
import numpy as np
import torch
import torch.nn as nn
from pruners.utils import *
import numpy.linalg as la
import numba as nb
from time import time
from sklearn.utils import extmath
from collections import namedtuple
import warnings
import copy
import time
from numba.core.errors import NumbaDeprecationWarning, \
    NumbaPendingDeprecationWarning, NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
from numba import prange
from torch.utils.data import DataLoader
import L0_card
import os


def generate_weight(model_name):
    
    if model_name == 'ResNetCifar':
        p = 268336
        para_list = [432,2304,2304,2304,2304,2304,2304,4608,9216,9216,9216,9216,9216,18432,36864,36864,36864,36864,36864,640]
        op_list = [1024,1024,1024,1024,1024,1024,1024, 256,256,256,256,256,256,64,64,64,64,64,64,1]
    elif model_name == 'MobileNet':
        p = 4209088
        para_list =[864,288,2048,576,8192,1152,16384,1152,32768,2304,65536,2304,131072,4608,
                    262144,4608,262144,4608,262144,4608,262144,4608,262144,4608,524288,9216,1048576,1024000]
        op_list = [12544.0,12544.0,12544.0,3136.0,3136.0,3136.0,3136.0,784.0,784.0,784.0,784.0,196.0,196.0,196.0,196.0,196.0,196.0,196.0,196.0,196.0,
                196.0,196.0,196.0,49.0,49.0,49.0,49.0,1.0]
    elif model_name == 'ResNet':
        p = 22734016
        para_list = [9408, 4096, 36864, 16384,  16384, 36864, 16384, 16384, 36864, 
             16384, 32768, 147456, 65536 , 65536, 147456, 65536, 65536, 147456, 
             65536, 65536, 147456, 65536, 131072, 589824, 262144, 262144, 589824,
            262144,262144, 589824,262144,262144, 589824,262144,262144, 589824, 262144,
             262144, 589824, 262144,524288,2359296,1048576,1048576,2359296,1048576,
             1048576,2359296,1048576, 2048000]
        op_list = [12544.0,  3136,  3136,  3136,  3136,  3136,  3136,  3136,  3136,
        3136,  3136,   784,   784,   784,   784,   784,   784,   784,
         784,   784,   784,   784,   784,   196,   196,   196,   196,
         196,   196,   196,   196,   196,   196,   196,   196,   196,
         196,   196,   196,   196,   196,    49,    49,    49,    49,
          49,    49,    49,    49,     1]
        
    elif model_name == 'ResNetdown':
        p = 25502912
        para_list = [9408, 4096, 36864, 16384,  16384, 16384, 36864, 16384, 16384, 36864, 
             16384, 32768, 147456, 65536 , 131072, 65536, 147456, 65536, 65536, 147456, 
             65536, 65536, 147456, 65536, 131072, 589824, 262144, 524288, 262144, 589824,
            262144,262144, 589824,262144,262144, 589824,262144,262144, 589824, 262144,
             262144, 589824, 262144,524288,2359296,1048576,2097152,1048576,2359296,1048576,
             1048576,2359296,1048576, 2048000]
        op_list = [12544.0,  3136,  3136,  3136,  3136,  3136,  3136,  3136,  3136, 3136,
           3136,  3136,   784,   784,   784,   784,   784,   784,   784, 784,
           784,   784,   784,   784,   784,   196,   196,   196,   196, 196,
           196,   196,   196,   196,   196,   196,   196,   196,   196, 196,
           196,   196,   196,   196,    49,    49,    49,    49,  49,    49,
           49,    49,    49,     1]
    elif model_name == 'wideresnet':
        p = 36461232
        para_list = [432, 23040, 230400, 2560, 230400, 230400, 230400, 230400, 230400, 230400,
            460800, 921600, 51200, 921600, 921600, 921600,921600, 921600, 921600, 1843200, 
             3686400, 204800, 3686400, 3686400, 3686400, 3686400, 3686400, 3686400, 6400]
        op_list = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
           1024, 256, 256, 256, 256, 256, 256, 256, 256, 256,
            64, 64, 64, 64, 64, 64, 64, 64, 1]
    elif model_name == 'wideresnet2':
        p = 36518832
        para_list = [432, 23040, 230400, 2560, 230400, 230400, 230400, 230400, 230400, 230400,
            460800, 921600, 51200, 921600, 921600, 921600,921600, 921600, 921600, 1843200, 
             3686400, 204800, 3686400, 3686400, 3686400, 3686400, 3686400, 3686400, 64000]
        op_list = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
           1024, 256, 256, 256, 256, 256, 256, 256, 256, 256,
            64, 64, 64, 64, 64, 64, 64, 64, 1]
    
    f_w = np.zeros(p)
    cur_id  = 0
    for i in range(len(para_list)):
        f_w[cur_id:cur_id+para_list[i]] = op_list[i] * np.ones(para_list[i])
        cur_id += para_list[i]
        
    return f_w

def get_search(v_w, model_name):
    
    if model_name == "ResNetCifar":
        para_list = [432,2304,2304,2304,2304,2304,2304,4608,9216,9216,9216,9216,9216,18432,36864,36864,36864,36864,36864,640]
    elif model_name == "MobileNet":
        para_list =[864,288,2048,576,8192,1152,16384,1152,32768,2304,65536,2304,131072,4608,
                    262144,4608,262144,4608,262144,4608,262144,4608,262144,4608,524288,9216,1048576,1024000]
    elif model_name == "ResNet":
        para_list = [9408, 4096, 36864, 16384,  16384, 36864, 16384, 16384, 36864, 
             16384, 32768, 147456, 65536 , 65536, 147456, 65536, 65536, 147456, 
             65536, 65536, 147456, 65536, 131072, 589824, 262144, 262144, 589824,
            262144,262144, 589824,262144,262144, 589824,262144,262144, 589824, 262144,
             262144, 589824, 262144,524288,2359296,1048576,1048576,2359296,1048576,
             1048576,2359296,1048576, 2048000]
    elif model_name == "ResNetdown":
        para_list = [9408, 4096, 36864, 16384,  16384, 16384, 36864, 16384, 16384, 36864, 
             16384, 32768, 147456, 65536 , 131072, 65536, 147456, 65536, 65536, 147456, 
             65536, 65536, 147456, 65536, 131072, 589824, 262144, 524288, 262144, 589824,
            262144,262144, 589824,262144,262144, 589824,262144,262144, 589824, 262144,
             262144, 589824, 262144,524288,2359296,1048576,2097152,1048576,2359296,1048576,
             1048576,2359296,1048576, 2048000]
    elif model_name == 'wideresnet':
        para_list = [432, 23040, 230400, 2560, 230400, 230400, 230400, 230400, 230400, 230400,
            460800, 921600, 51200, 921600, 921600, 921600,921600, 921600, 921600, 1843200, 
             3686400, 204800, 3686400, 3686400, 3686400, 3686400, 3686400, 3686400, 6400]
    elif model_name == 'wideresnet2':
        para_list = [432, 23040, 230400, 2560, 230400, 230400, 230400, 230400, 230400, 230400,
            460800, 921600, 51200, 921600, 921600, 921600,921600, 921600, 921600, 1843200, 
             3686400, 204800, 3686400, 3686400, 3686400, 3686400, 3686400, 3686400, 64000]
        
    sorted_list = []
    cur_id  = 0
    for i in range(len(para_list)):
        sorted_list.append(np.sort(v_w[cur_id:cur_id+para_list[i]]))
        cur_id += para_list[i]
    
    return sorted_list

def new_sort(lambda_2, sorted_list, model_name):
    
    if model_name == "ResNetCifar":
        op_list = [1024,1024,1024,1024,1024,1024,1024, 256,256,256,256,256,256,64,64,64,64,64,64,1]
    elif model_name == "MobileNet":
        op_list = [12544.0,12544.0,12544.0,3136.0,3136.0,3136.0,3136.0,784.0,784.0,784.0,784.0,196.0,196.0,196.0,196.0,196.0,196.0,196.0,196.0,196.0,
                196.0,196.0,196.0,49.0,49.0,49.0,49.0,1.0]
    elif model_name == "ResNet":
        op_list = [12544.0,  3136,  3136,  3136,  3136,  3136,  3136,  3136,  3136,
        3136,  3136,   784,   784,   784,   784,   784,   784,   784,
         784,   784,   784,   784,   784,   196,   196,   196,   196,
         196,   196,   196,   196,   196,   196,   196,   196,   196,
         196,   196,   196,   196,   196,    49,    49,    49,    49,
          49,    49,    49,    49,     1]
    elif model_name == "ResNetdown":
        op_list = [12544.0,  3136,  3136,  3136,  3136,  3136,  3136,  3136,  3136, 3136,
           3136,  3136,   784,   784,   784,   784,   784,   784,   784, 784,
           784,   784,   784,   784,   784,   196,   196,   196,   196, 196,
           196,   196,   196,   196,   196,   196,   196,   196,   196, 196,
           196,   196,   196,   196,    49,    49,    49,    49,  49,    49,
           49,    49,    49,     1]
    elif model_name == 'wideresnet':
        op_list = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
           1024, 256, 256, 256, 256, 256, 256, 256, 256, 256,
            64, 64, 64, 64, 64, 64, 64, 64, 1]
    elif model_name == 'wideresnet2':
        op_list = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
           1024, 256, 256, 256, 256, 256, 256, 256, 256, 256,
            64, 64, 64, 64, 64, 64, 64, 64, 1]
        
    sort_res = copy.deepcopy([sorted_list[i] + lambda_2 * op_list[i]  for i in range(len(sorted_list))])
    while len(sort_res) > 1:
        
        arg_len = np.argsort([len(layer_sort) for layer_sort in sort_res])
        i1, i2 = arg_len[0], arg_len[1]
        cur_sort = np.concatenate((sort_res[i1] ,sort_res[i2]))
        cur_sort.sort(kind='mergesort')
        if i1 < i2:
            sort_res.pop(i2)
            sort_res.pop(i1)
        else:
            sort_res.pop(i1)
            sort_res.pop(i2)
        sort_res.append(cur_sort)
    return np.copy(sort_res[0])


def flop_proj(v_w, f_w, S, F, stop_acc, model_name):
    
    
    sorted_list = get_search(v_w, model_name)
    
    lam_lb = 0
    lam_ub = -np.min(v_w/f_w)
    gold_ratio = (np.sqrt(5)-1)/2
    
    tot_iter = 0
    lam_1 = lam_lb + (lam_ub-lam_lb) * (1-gold_ratio)
    lam_2 = lam_lb + (lam_ub-lam_lb) * gold_ratio
    #vec_1 = np.sort(v_w + lam_1 * f_w)
    #vec_2 = np.sort(v_w + lam_2 * f_w)
    vec_1 = new_sort(lam_1, sorted_list, model_name)
    vec_2 = new_sort(lam_2, sorted_list, model_name)
    ome_1 = np.maximum(-vec_1[S],0)
    ome_2 = np.maximum(-vec_2[S],0)
    
    obj_1 = S * ome_1 + F * lam_1 + np.sum( np.maximum(-vec_1-ome_1,0) )
    obj_2 = S * ome_2 + F * lam_2 + np.sum( np.maximum(-vec_2-ome_2,0) )
    
    while lam_ub - lam_lb > stop_acc:
    
        tot_iter  += 1
        if obj_1 > obj_2:
        
            lam_lb = lam_1
            lam_1 = lam_2
            lam_2 = lam_lb + (lam_ub-lam_lb) * gold_ratio
        
            obj_1 = obj_2
            #vec_2 = np.sort(v_w + lam_2 * f_w)
            vec_2 = new_sort(lam_2, sorted_list, model_name)
            ome_2 = np.maximum(-vec_2[S], 0)
            obj_2 = S * ome_2 + F * lam_2 + np.sum( np.maximum(-vec_2-ome_2,0) )
        
        else:
        
            lam_ub = lam_2
            lam_2 = lam_1
            lam_1 = lam_lb + (lam_ub-lam_lb) * (1-gold_ratio)
        
            obj_2 = obj_1
            #vec_1 = np.sort(v_w + lam_1 * f_w)
            vec_1 = new_sort(lam_1, sorted_list, model_name)
            ome_1 = np.maximum(-vec_1[S], 0)
            obj_1 = S * ome_1 + F * lam_1 + np.sum( np.maximum(-vec_1-ome_1,0))
    
    
    vec_0 = v_w + (lam_1+lam_2)/2 * f_w
    idx_w = np.argsort(vec_0)
    idx_s = np.searchsorted(np.cumsum(f_w[idx_w]), F, side ='right')
    thres = np.minimum(vec_0[idx_w[np.minimum(idx_s,S)]], 0)
    ind_mask = (vec_0) < thres
    
    print((lam_1+lam_2)/2, np.maximum(-np.sort(vec_0)[S],0))
    
    return ind_mask, tot_iter


def WF_solve(X, beta, w_bar, lambda2):
    
    n, p = X.shape
    nonact_idx = np.where(beta==0)[0]
    tmp_inv = np.linalg.inv(np.eye(n)+X@(X.T)/(2*lambda2))
    Xmulinv = X.T@tmp_inv
    w_minus = np.zeros(p)
    for i in nonact_idx:
        Hinv = 1 / (2*lambda2) - (Xmulinv[i:i+1,:]@X[:,i]) / (4*lambda2**2)
        w_minus[i] = w_bar[i] / Hinv
        
    w_invmin = w_minus / (2*lambda2) - \
            X.T@np.linalg.solve(np.eye(n)+X@(X.T)/(2*lambda2),X@w_minus)/(4*lambda2**2)
    w_pruned = w_bar - w_invmin
    w_pruned[nonact_idx] = 0
    return w_pruned



def FLOP_solve(y,X,w1,k,flop_k,alpha,lambda2, beta_tilde2, model_name, block_diag, solve_method, sol_opt):
    
    
    n, p = X.shape
    if block_diag == None:
        block_diag = [0,p]
    # generate support
    v_w = -w1**2
    f_w = generate_weight(model_name)
    beta = np.copy(w1)
    ind_mask, tot_iter = flop_proj(v_w, f_w, k, flop_k, 1e-12, model_name)
    s_cur = len(np.where(ind_mask)[0])
    beta *= ind_mask
    
    if solve_method == "MP":
        w_pruned = np.copy(beta)
            
    elif solve_method == "WF":
            
        w_advpre = np.copy(w1)
        w_pruned = np.zeros(p)
        for jj in range(len(block_diag)-1):
            w_pruned[block_diag[jj]:block_diag[jj+1]] = WF_solve(X[:,block_diag[jj]:block_diag[jj+1]], 
                                beta[block_diag[jj]:block_diag[jj+1]], w_advpre[block_diag[jj]:block_diag[jj+1]], lambda2)
        
    elif solve_method == "BS":    
                 
        w_pruned, _, _, sol_time = L0_card.Heuristic_LSBlock(np.copy(w1),
                X,beta,len(np.where(beta)[0]),alpha=alpha, lambda1=0.,lambda2=lambda2, beta_tilde1=np.zeros(p), 
                beta_tilde2=np.copy(w1), M=np.inf,use_prune = False, per_idx=None, num_block=None, 
                block_list=block_diag, split_type=1)
        
    elif solve_method == "IHT":    
        
        num_flopiter = 5
        proj_iter = np.linspace(0,sol_opt["iht_iter"],num_flopiter).astype("int")
        w_pruned = np.copy(beta)
        
        for pi in range(num_flopiter-1):
            w_pruned,_,_,_,_,_ = L0_card.Active_IHTCDLS_PP(y,X,w_pruned,k,
                    alpha=alpha,lambda1=0.,lambda2=lambda2, beta_tilde1=np.zeros(p),beta_tilde2=np.copy(w1), L=None, M=np.inf, 
                    iht_max_itr=proj_iter[pi+1]-proj_iter[pi], ftol=1e-8, act_max_itr=10,buget=None,kimp=2.,act_itr=1,cd_itr=0,ctol=1e-4, 
                    sea1_max_itr=5, sea2_max_itr=2)
            
         
            if np.sum(f_w * (w_pruned!=0)) > flop_k:
                
                idx_w = np.argsort(-np.abs(w_pruned)/f_w)
                idx_s = np.searchsorted(np.cumsum(f_w[idx_w]), flop_k, side ='right')
                w_pruned[idx_w[idx_s:]] = 0
        
     
    
    obj = 0.5 * np.linalg.norm(y-X@w_pruned)**2 + lambda2 *  np.linalg.norm(w1-w_pruned)**2 
    
    return w_pruned, obj


def CAIE1_solve(y,X,w1,k,flop_k,alpha,lambda2, beta_tilde2, model_name, block_diag, solve_method, sol_opt):
    
    
    n, p = X.shape
    if block_diag == None:
        block_diag = [0,p]
        

    f_w = generate_weight(model_name)
    
    tmp_inv = np.linalg.inv(np.eye(n)+X@(X.T)/(2*lambda2))
    Xmulinv = X.T@tmp_inv
    
    Hinv = 1 / (2*lambda2) - np.sum(Xmulinv. T * X, axis = 0) / (4*lambda2**2)
    
    idx_w = np.argsort(-np.abs(w1)**2 / Hinv / (0.0005*f_w+1)) # resnet20 -- 0.015 # 0.0001
    idx_s = np.searchsorted(np.cumsum(f_w[idx_w]), flop_k)
    ind_mask = np.zeros_like(w1)
    ind_mask[idx_w[:idx_s]] = 1
    
    w_pruned = np.copy(w1)
    w_pruned *= ind_mask
        
    obj = 0.5 * np.linalg.norm(y-X@w_pruned)**2 + lambda2 *  np.linalg.norm(w1-w_pruned)**2 
    
    return w_pruned, obj


def WFbeta_solve(y,X,w1,k,flop_k,alpha,lambda2, beta_tilde2, model_name, block_diag, solve_method, sol_opt):
    
    
    n, p = X.shape
    if block_diag == None:
        block_diag = [0,p]
        
    WFb = 0.3
    f_w = generate_weight(model_name)
    
    idx_w = np.argsort(-np.abs(w1) / f_w**WFb)
    idx_s = np.searchsorted(np.cumsum(f_w[idx_w]), flop_k)
    ind_mask = np.zeros_like(w1)
    ind_mask[idx_w[:idx_s]] = 1
    
    beta = np.copy(w1)
    beta *= ind_mask
    
    w_advpre = np.copy(w1)
    w_pruned = np.zeros(p)
    for jj in range(len(block_diag)-1):
        w_pruned[block_diag[jj]:block_diag[jj+1]] = WF_solve(X[:,block_diag[jj]:block_diag[jj+1]], 
                                beta[block_diag[jj]:block_diag[jj+1]], w_advpre[block_diag[jj]:block_diag[jj+1]], lambda2)
        
    obj = 0.5 * np.linalg.norm(y-X@w_pruned)**2 + lambda2 *  np.linalg.norm(w1-w_pruned)**2 
    
    return w_pruned, obj


def WFbeta2_solve(y,X,w1,k,flop_k,alpha,lambda2, beta_tilde2, model_name, block_diag, solve_method, sol_opt):
    
    
    n, p = X.shape
    if block_diag == None:
        block_diag = [0,p]
        
    WFb = 0.3
    f_w = generate_weight(model_name)
    
    
    tmp_inv = np.linalg.inv(np.eye(n)+X@(X.T)/(2*lambda2))
    Xmulinv = X.T@tmp_inv
    
    Hinv = 1 / (2*lambda2) - np.sum(Xmulinv. T * X, axis = 0) / (4*lambda2**2)
        
    
    idx_w = np.argsort(-np.abs(w1)**2 / Hinv / f_w**WFb)
    idx_s = np.searchsorted(np.cumsum(f_w[idx_w]), flop_k)
    ind_mask = np.zeros_like(w1)
    ind_mask[idx_w[:idx_s]] = 1
    
    beta = np.copy(w1)
    beta *= ind_mask
    
    w_advpre = np.copy(w1)
    w_pruned = np.zeros(p)
    for jj in range(len(block_diag)-1):
        w_pruned[block_diag[jj]:block_diag[jj+1]] = WF_solve(X[:,block_diag[jj]:block_diag[jj+1]], 
                                beta[block_diag[jj]:block_diag[jj+1]], w_advpre[block_diag[jj]:block_diag[jj+1]], lambda2)
        
    obj = 0.5 * np.linalg.norm(y-X@w_pruned)**2 + lambda2 *  np.linalg.norm(w1-w_pruned)**2 
    
    return w_pruned, obj