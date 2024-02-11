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


def retrive_model(model = "resnet20"):
    
    if model == "resnet20":
        arch = "resnet20"
        dset_path = './datasets'
        model,train_dataset,test_dataset,criterion,modules_to_prune = model_factory(arch,dset_path, True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=10,pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True,num_workers=10,pin_memory=True)
        model.to(device)
        model.train()
        compute_acc(model,DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers=10,pin_memory=True),device)
        model.eval()
        
    elif model == "mobile":
        arch = "mobilenetv1"
        #print(os.environ)
        #dset_path = '/run/user/62136/loopmnt1'+'/raw'
        dset_path = os.environ['IMAGENET_PATH']+'/raw'
        model,train_dataset,test_dataset,criterion,modules_to_prune = model_factory(arch,dset_path, True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,num_workers=10,pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=10,pin_memory=True)
        model = model.to(device)
        model.eval()
    
    elif model == "resnet50":
        dset_path = os.environ['IMAGENET_PATH']+'/raw'
        model,train_dataset,test_dataset,criterion,modules_to_prune = model_factory('resnet50',dset_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,num_workers=10,pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=50, shuffle=False,num_workers=10,pin_memory=True)
        model.eval()
        model.to(device)
        
        
    return model, train_dataloader, test_dataloader, criterion, modules_to_prune
        
def generate_weight(model = "resnet20"):
    
    if model == "resnet20":
        p = 268336
        para_list = [432,2304,2304,2304,2304,2304,2304,4608,9216,9216,9216,9216,9216,18432,36864,36864,36864,36864,36864,640]
        op_list = [1024,1024,1024,1024,1024,1024,1024, 256,256,256,256,256,256,64,64,64,64,64,64,1]
    elif model == "mobile":
        p = 4209088
        para_list =[864,288,2048,576,8192,1152,16384,1152,32768,2304,65536,2304,131072,4608,
                    262144,4608,262144,4608,262144,4608,262144,4608,262144,4608,524288,9216,1048576,1024000]
        op_list = [12544.0,12544.0,12544.0,3136.0,3136.0,3136.0,3136.0,784.0,784.0,784.0,784.0,196.0,196.0,196.0,196.0,196.0,196.0,196.0,196.0,196.0,
                196.0,196.0,196.0,49.0,49.0,49.0,49.0,1.0]
    elif model == "resnet50":
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
    
    f_w = np.zeros(p)
    cur_id  = 0
    for i in range(len(para_list)):
        f_w[cur_id:cur_id+para_list[i]] = op_list[i] * np.ones(para_list[i])
        cur_id += para_list[i]
        
    return f_w

def cal_grad(model_new, w_new, num_grad, modules_to_prune, train_dataloader, criterion):
    
    device = "cuda"
    p = len(w_new)
    set_pvec(w_new, model_new, modules_to_prune, "cuda")
    X = torch.zeros((num_grad, p), device='cpu')
    for i, batch in enumerate(train_dataloader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        loss = criterion(model_new(x), y)
        loss.backward()
        X[i] = get_gvec(model_new, modules_to_prune).to('cpu')
        zero_grads(model_new)
        if (i + 1) % num_grad == 0:
            break
    X = X.numpy().astype(np.float64)
    
    return X

def cal_acc(w_pruned, f_w, model, modules_to_prune, test_dataloader):
    
    model_new = copy.deepcopy(model)
    model_new = model_new.to("cuda")
    model_new.eval()
    set_pvec(w_pruned, model_new, modules_to_prune,"cuda")
    
    nnz = len(np.where(w_pruned != 0)[0])
    nnflop = np.sum((w_pruned !=0).astype("float64") * f_w)
    return compute_acc(model_new,test_dataloader,"cuda"), nnz, nnflop

def iht_multi(v_w, f_w, S, F, stop_acc, model = "resnet20"):
    
    
    sorted_list = get_search(v_w, model)
    
    lam_lb = 0
    lam_ub = -np.min(v_w/f_w)
    gold_ratio = (np.sqrt(5)-1)/2
    
    tot_iter = 0
    lam_1 = lam_lb + (lam_ub-lam_lb) * (1-gold_ratio)
    lam_2 = lam_lb + (lam_ub-lam_lb) * gold_ratio
    vec_1 = np.sort(v_w + lam_1 * f_w)
    vec_2 = np.sort(v_w + lam_2 * f_w)
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
            vec_2 = new_sort(lam_2, sorted_list, model)
            ome_2 = np.maximum(-vec_2[S], 0)
            obj_2 = S * ome_2 + F * lam_2 + np.sum( np.maximum(-vec_2-ome_2,0) )
        
        else:
        
            lam_ub = lam_2
            lam_2 = lam_1
            lam_1 = lam_lb + (lam_ub-lam_lb) * (1-gold_ratio)
        
            obj_2 = obj_1
            #vec_1 = np.sort(v_w + lam_1 * f_w)
            vec_1 = new_sort(lam_1, sorted_list, model)
            ome_1 = np.maximum(-vec_1[S], 0)
            obj_1 = S * ome_1 + F * lam_1 + np.sum( np.maximum(-vec_1-ome_1,0))
    
    vec_0 = v_w + (lam_1+lam_2)/2 * f_w
    thres = np.minimum(np.sort(vec_0)[S], 0)
    ind_mask = (vec_0) < thres
    
    return ind_mask, tot_iter

def get_search(v_w, model = "resnet20"):
    
    if model == "resnet20":
        para_list = [432,2304,2304,2304,2304,2304,2304,4608,9216,9216,9216,9216,9216,18432,36864,36864,36864,36864,36864,640]
    elif model == "mobile":
        para_list =[864,288,2048,576,8192,1152,16384,1152,32768,2304,65536,2304,131072,4608,
                    262144,4608,262144,4608,262144,4608,262144,4608,262144,4608,524288,9216,1048576,1024000]
    elif model == "resnet50":
        para_list = [9408, 4096, 36864, 16384,  16384, 36864, 16384, 16384, 36864, 
             16384, 32768, 147456, 65536 , 65536, 147456, 65536, 65536, 147456, 
             65536, 65536, 147456, 65536, 131072, 589824, 262144, 262144, 589824,
            262144,262144, 589824,262144,262144, 589824,262144,262144, 589824, 262144,
             262144, 589824, 262144,524288,2359296,1048576,1048576,2359296,1048576,
             1048576,2359296,1048576, 2048000]
        
    sorted_list = []
    cur_id  = 0
    for i in range(len(para_list)):
        sorted_list.append(np.sort(v_w[cur_id:cur_id+para_list[i]]))
        cur_id += para_list[i]
    
    return sorted_list
        
def new_sort(lambda_2, sorted_list, model):
    
    if model == "resnet20":
        op_list = [1024,1024,1024,1024,1024,1024,1024, 256,256,256,256,256,256,64,64,64,64,64,64,1]
    elif model == "mobile":
        op_list = [12544.0,12544.0,12544.0,3136.0,3136.0,3136.0,3136.0,784.0,784.0,784.0,784.0,196.0,196.0,196.0,196.0,196.0,196.0,196.0,196.0,196.0,
                196.0,196.0,196.0,49.0,49.0,49.0,49.0,1.0]
    elif model == "resnet50":
        op_list = [12544.0,  3136,  3136,  3136,  3136,  3136,  3136,  3136,  3136,
        3136,  3136,   784,   784,   784,   784,   784,   784,   784,
         784,   784,   784,   784,   784,   196,   196,   196,   196,
         196,   196,   196,   196,   196,   196,   196,   196,   196,
         196,   196,   196,   196,   196,    49,    49,    49,    49,
          49,    49,    49,    49,     1]
        
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

def multi_stage(flop_list, sparsity_list, cons_type, init_type, model, w_bar, num_grad, modules_to_prune, train_dataloader, test_dataloader, criterion, flopmodel, seed, block_size, sol_opt, solve_method):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    st_init = time.time()
    w_pruned = np.copy(w_bar)
    f_w = generate_weight(flopmodel)
    p = len(w_pruned)
    sparsity_list = [int(p*cc) for cc in sparsity_list]
    model_new = copy.deepcopy(model)
    time_c = 0
    time_p = 0
    for jjj in range(len(flop_list)):
    
        if solve_method != "MP":
            X = cal_grad(model_new, w_pruned, num_grad, modules_to_prune, train_dataloader, criterion)  
            ynew = X@w_pruned
        beta = np.copy(w_pruned)
        s_cur = len(np.where(beta != 0))
        
        if cons_type == "S":
            
            if init_type == "S":
                s_cur = sparsity_list[jjj]
                argidx = np.argsort(-np.abs(beta)**2)
                beta[argidx[s_cur:]] = 0 
            else:
                s_cur = sparsity_list[jjj]
                argidx = np.argsort(-np.abs(beta)**2/f_w)
                beta[argidx[s_cur:]] = 0 
                
        elif cons_type == "F":
            
            if init_type == "S":
                f_cur = flop_list[jjj]
                argidx = np.argsort(-np.abs(beta)**2)
                flop_cum = np.cumsum(f_w[argidx[:]])
                s_cur = np.searchsorted(flop_cum, f_cur, side="right")
                beta[argidx[s_cur:]] = 0 
            else:
                f_cur = flop_list[jjj]
                argidx = np.argsort(-np.abs(beta)**2/f_w)
                flop_cum = np.cumsum(f_w[argidx[:]])
                s_cur = np.searchsorted(flop_cum, f_cur, side="right")
                beta[argidx[s_cur:]] = 0 
                
        else:
            st = time.time()
            
            v_w = -beta**2
            ind_mask, tot_iter = iht_multi(v_w, f_w, sparsity_list[jjj], flop_list[jjj], 1e-12, flopmodel)
            s_cur = len(np.where(ind_mask)[0])
            beta *= ind_mask
            time_c += time.time()-st
            print("Time cons:", time.time()-st)
        
        st = time.time()
        if solve_method == "MP":
            
            w_pruned = np.copy(beta)
            
        elif solve_method == "WF":
            

            ignore_bias = True
            # generate blocklist (just once for multi-stages)
            i_w = 0
            block_diag = [0]
            for name, param in model_new.named_parameters():
                layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
                if ignore_bias == True and param_name == 'bias':
                    continue
                if (not modules_to_prune is None) and (not name in modules_to_prune):
                    continue
                param_size = np.prod(param.shape)
     
                if param_size <block_size:
                    block_diag.append(i_w+param_size)
                else:
                    num_block = int(param_size/block_size)
                    block_subdiag = list(range(i_w,i_w+param_size+1,int(param_size/num_block))) 
                    block_subdiag[-1] = i_w+param_size
                    block_diag += block_subdiag   
                i_w += param_size
            w_advpre = np.copy(w_pruned)
            w_pruned = np.zeros(p)
            for jj in range(len(block_diag)-1):
                w_pruned[block_diag[jj]:block_diag[jj+1]] = WF_solve(X[:,block_diag[jj]:block_diag[jj+1]], 
                                            beta[block_diag[jj]:block_diag[jj+1]], w_advpre[block_diag[jj]:block_diag[jj+1]], sol_opt["lambda2"])
            
                
                
        elif solve_method == "BS":    
           

            ignore_bias = True
            # generate blocklist (just once for multi-stages)
            i_w = 0
            block_diag = [0]
            for name, param in model_new.named_parameters():
                layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
                if ignore_bias == True and param_name == 'bias':
                    continue
                if (not modules_to_prune is None) and (not name in modules_to_prune):
                    continue
                param_size = np.prod(param.shape)
     
                if param_size <block_size:
                    block_diag.append(i_w+param_size)
                else:
                    num_block = int(param_size/block_size)
                    block_subdiag = list(range(i_w,i_w+param_size+1,int(param_size/num_block))) 
                    block_subdiag[-1] = i_w+param_size
                    block_diag += block_subdiag   
                i_w += param_size
                    
            w_pruned, _, _, sol_time = L0_card.Heuristic_LSBlock(np.copy(w_pruned),
                X,beta,s_cur,alpha=np.zeros(p), lambda1=0.,lambda2=sol_opt["lambda2"], beta_tilde1=np.zeros(p), 
                beta_tilde2=np.copy(w_pruned), M=np.inf,use_prune = True, per_idx=None, num_block=None, 
                block_list=block_diag, split_type=1)
        
        elif solve_method == "IHT":    
           
            ignore_bias = True
            # generate blocklist (just once for multi-stages)
            i_w = 0
            block_diag = [0]
            for name, param in model_new.named_parameters():
                layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
                if ignore_bias == True and param_name == 'bias':
                    continue
                if (not modules_to_prune is None) and (not name in modules_to_prune):
                    continue
                param_size = np.prod(param.shape)
     
                if param_size <block_size:
                    block_diag.append(i_w+param_size)
                else:
                    num_block = int(param_size/block_size)
                    block_subdiag = list(range(i_w,i_w+param_size+1,int(param_size/num_block))) 
                    block_subdiag[-1] = i_w+param_size
                    block_diag += block_subdiag   
                i_w += param_size
            w_advpre = np.copy(w_pruned)
            w_pruned = np.zeros(p)
            if sol_opt["use_alpha"]:
                alpha = X.sum(axis=0)
            else:
                alpha = np.zeros(p)
            for jj in range(len(block_diag)-1):
                w_ind = beta[block_diag[jj]:block_diag[jj+1]]
                w_ind2 = w_advpre[block_diag[jj]:block_diag[jj+1]]
                X_ind = X[:,block_diag[jj]:block_diag[jj+1]]
                y_ind = X_ind@w_ind2
                k_ind = np.sum(np.abs(w_ind) > 0)
                p_ind = block_diag[jj+1] - block_diag[jj]
                if k_ind == 0:
                    w_pruned[block_diag[jj]:block_diag[jj+1]] = np.zeros(p_ind)
                    continue
                w_pruned[block_diag[jj]:block_diag[jj+1]],_,_,_,_,_ = L0_card.Active_IHTCDLS_PP(y_ind,X_ind,w_ind,k_ind,
                    alpha=alpha[block_diag[jj]:block_diag[jj+1]],lambda1=0.,lambda2=sol_opt["lambda2"], beta_tilde1=np.zeros(p_ind),beta_tilde2=w_ind2, L=None, M=np.inf, 
                    iht_max_itr=sol_opt["iht_iter"], ftol=1e-8, act_max_itr=10,buget=None,kimp=2.,act_itr=1,cd_itr=sol_opt["cd_iter"],ctol=1e-4, 
                    sea1_max_itr=5, sea2_max_itr=2)
                    
        acc, spr, flo = cal_acc(w_pruned, f_w, model, modules_to_prune, test_dataloader)
        time_p += time.time() - st
        print("Sparsity is ",1-spr/p)
        print("Flop is ",flo / 10**6,"M")
        print("Acc is ",acc)
    time_t = time.time() - st_init
        
    return w_pruned, acc , 1-spr/p, flo, (time_t, time_c, time_p)

