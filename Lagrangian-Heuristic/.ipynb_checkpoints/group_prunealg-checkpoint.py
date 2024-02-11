from errno import ENETUNREACH
import numpy as np
import torch
import torch.nn as nn
import numpy.linalg as la
from time import time
from sklearn.utils import extmath
from sklearn.utils.extmath import randomized_svd
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
from pruners.utils import *
from scipy.spatial import distance

def get_layerobj(model, modules_to_prune, seed, num_samples, train_dataset, w_in, w_out, si_list, drop_down = True, batch_size = 8):
    
    def conv_hook(self, input, output):
    
        input_buffer.append(input[0])
        output_buffer.append(output)

    def linear_hook(self, input, output):
    
        input_buffer.append(input[0])
        output_buffer.append(output)
        
    def conv_hook2(self, input, output):
    
        input_buffer2.append(input[0])
        output_buffer2.append(output)

    def linear_hook2(self, input, output):
    
        input_buffer2.append(input[0])
        output_buffer2.append(output)
    
    def foo(net, target_name, save_buffer = 1, name=''):
        
        #print("name is",name)
        children = list(net.named_children())
        if not children:
            if name + ".weight" == target_name:
                if isinstance(net, torch.nn.Conv2d):
                    if save_buffer == 1:
                        net.register_forward_hook(conv_hook)
                    else:
                        net.register_forward_hook(conv_hook2)
                if isinstance(net, torch.nn.Linear):
                    if save_buffer == 1:
                        net.register_forward_hook(linear_hook)
                    else:
                        net.register_forward_hook(linear_hook2)
                    
        for child_name, child in children:
            if name:
                foo(child, target_name, save_buffer, name="{}.{}".format(name, child_name))
            else:
                foo(child, target_name, save_buffer, child_name)
            
    
    def find_layers(module, layers=[nn.Conv2d, nn.Linear], layer_sp = []):
    
        if type(module) in layers:
            layer_sp.append(module)
            return layer_sp
        for child_name, child in module.named_children():
            if child_name == "downsample" and drop_down:
                continue
            find_layers(child, layers=layers, layer_sp=layer_sp)
        return layer_sp
    
    size_list = []
    ignore_bias = True
    for name, param in model.named_parameters():
        layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
        if ignore_bias == True and param_name == 'bias':
            continue
        if (not modules_to_prune is None) and (not name in modules_to_prune):
            continue
        
        param_size = np.prod(param.shape)
        size_list.append(param.shape)
    
    layersp = find_layers(model)
    
    hess_list = []
    xty_list = []
    for si in si_list:
        sum_w = np.prod(size_list[si][1:])
        hess_list.append(np.zeros((sum_w,sum_w)))
        xty_list.append(np.zeros((sum_w,size_list[si][0])))
    
    # collect data
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=10,pin_memory=True)
            
    for ci, batch in enumerate(train_dataloader):
    
        xdata, ydata = batch
        xdata = xdata.to("cuda")
        model_new = copy.deepcopy(model)
        model_new2 = copy.deepcopy(model)
            
        set_pvec(w_in, model_new, modules_to_prune,"cuda")
        set_pvec(w_out, model_new2, modules_to_prune,"cuda")
        
        for csi in range(len(si_list)):
            
            si = si_list[csi]
            target_name = modules_to_prune[si]
            
            input_buffer = []
            output_buffer = []
            
            input_buffer2 = []
            output_buffer2 = []
            
        
            foo(model_new,target_name,1)
            model_new(xdata)
        
            foo(model_new2,target_name,2)
            model_new2(xdata)

            
            unfold = nn.Unfold(
                layersp[si].kernel_size,
                dilation=layersp[si].dilation,
                padding=layersp[si].padding,
                stride=layersp[si].stride
            )
            
            
            inp = np.vstack([unfold(inss).permute([1, 0, 2]).flatten(1).to("cpu").detach().numpy() for inss in input_buffer])
            inp2 = np.vstack([unfold(inss).permute([1, 0, 2]).flatten(1).to("cpu").detach().numpy() for inss in input_buffer2])
            
            #inp = unfold(input_buffer[si])
            #inp = inp.permute([1, 0, 2])
            #inp = inp.flatten(1).to("cpu").detach().numpy()
            
            #inp2 = unfold(input_buffer2[si+len(size_list)])
            #inp2 = inp2.permute([1, 0, 2])
            #inp2 = inp2.flatten(1).to("cpu").detach().numpy()
            
            hess_list[csi] += inp @ inp.T / num_samples / batch_size
            
            i_st = np.sum([np.prod(size_list[tsi]) for tsi in range(si)])
            i_end = i_st + np.prod(size_list[si])   
            w_vec = w_out[i_st:i_end]
            w_var = w_vec.reshape(size_list[si][0],-1).T
            xty_list[csi] += (inp @ inp2.T) @ w_var  / num_samples / batch_size
            
        
        if (ci + 1) % num_samples == 0:
            break
            
    
    return hess_list, xty_list

def find_module(net, params_to_prune, name=''):

    children = list(net.named_children())
    if not children:
        if name+".weight" == params_to_prune:
            return True, net
        else:
            return False, None 
    for child_name, child in children:
        if name:
            output_flag, net = find_module(child, params_to_prune, name="{}.{}".format(name, child_name))
        else:
            output_flag, net = find_module(child, params_to_prune, name=child_name)
        if output_flag:
            return True, net
    return False, None

def find_all_module(net, params_to_prune, name='', prune_list = []):

    children = list(net.named_children())
    if not children:
        if name+".weight" in params_to_prune:
            prune_list.append(name+".weight")
            
    for child_name, child in children:
        if name:
            find_all_module(child, params_to_prune, name="{}.{}".format(name, child_name), prune_list=prune_list)
        else:
            find_all_module(child, params_to_prune, name=child_name, prune_list=prune_list)

    return prune_list

def get_blocks(model):
    
    # input copied model
    if model.name == 'ResNetCifar':
        
        block_list = []
        child_list = list(model.named_children())
        
        block_list.append(("",nn.Sequential(OrderedDict([
          ('conv1', child_list[0][1]),
          ('bn1', child_list[1][1]),
          ('relu', child_list[2][1])
          ]))))
        
        for i in range(3,6):
            for name, child in child_list[i][1].named_children():
                block_list.append((child_list[i][0]+"."+name,child))
                
        block_list.append(("",nn.Sequential(OrderedDict([('avgpool',child_list[6][1]),('flatten',nn.Flatten()),('fc',child_list[7][1])]))))
        
    if model.name == 'MobileNet':
        
        block_list = []
        child_list = list(model.named_children())
        child_name = child_list[0][0]
        child_list2 = list(child_list[0][1].named_children())
        for i in range(14):
            block_list.append((child_name+"."+child_list2[i][0],child_list2[i][1]))
            
        block_list.append(("",nn.Sequential(OrderedDict([('avgpool',child_list2[-1][1]),('flatten',nn.Flatten()),('fc',child_list[-1][1])]))))
        
    if model.name == 'ResNet':
        
        block_list = []
        child_list = list(model.named_children())
        
        block_list.append(("",nn.Sequential(OrderedDict([
          ('conv1', child_list[0][1]),
          ('bn1', child_list[1][1]),
          ('relu', child_list[2][1]),
          ('maxpool', child_list[3][1])
          ]))))
        
        for i in range(4,8):
            for name, child in child_list[i][1].named_children():
                block_list.append((child_list[i][0]+"."+name,child))
        
        block_list.append(("",nn.Sequential(OrderedDict([('avgpool',child_list[8][1]),('flatten',nn.Flatten()),('fc',child_list[9][1])]))))
            
        
    return block_list        

def weight_update(W, XTX, XTY):
    
    W_sol = np.zeros_like(W)
    nzi = np.nonzero(W[:,0])[0]
    XTX_sub = XTX[np.ix_(nzi,nzi)]
    XTY_sub = XTY[nzi,:]
    
    W_sol[nzi,:] = np.linalg.inv(XTX_sub)@ XTY_sub
    return W_sol, np.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) )

def weight_selection(W, XTX, XTY, num_cin, num_sp):
    
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    W = W.reshape(num_cin, ksize, num_cout)
    idx = np.argsort(np.sum(np.sum(np.abs(W),axis=2),axis=1))
    W[idx[:num_cin-num_sp],:,:] = 0
    W = W.reshape(totp, num_cout)

    return W, np.sum( -W * XTY + (1/2) * W * (XTX@W) )

def geomedian_selection(W, XTX, XTY, num_cin, num_sp, filter_ratio=0.2):
    
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    num_filter = int(num_sp * filter_ratio)
    num_dist = (num_cin-num_sp) 
    
    W = W.reshape(num_cin, ksize * num_cout)
    idx1 = np.argsort(np.linalg.norm(W,2,1))
    mask_filter = np.ones((num_cin,))
    mask_filter[idx1[:num_filter]] = 0
    
    idx2 = np.argsort(np.sum(distance.cdist(W, W, 'euclidean') * mask_filter,axis=1) + 1e20 * (1- mask_filter))
    W[idx2[:num_cin-num_sp],:] = 0
    W = W.reshape(totp, num_cout)

    return W, np.sum( -W * XTY + (1/2) * W * (XTX@W) )

def channel_selection(W, XTX, XTY, num_cin, num_sp):
    
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    W = W.reshape(num_cin, ksize, num_cout)
    idx = np.argsort(np.sum(np.sum(np.abs(W),axis=2),axis=1))
    W[idx[:num_cin-num_sp],:,:] = 0
    W = W.reshape(totp, num_cout)

    W_sol = np.zeros_like(W)
    nzi = np.nonzero(W[:,0])[0]
    XTX_sub = XTX[np.ix_(nzi,nzi)]
    XTY_sub = XTY[nzi,:]
    
    W_sol[nzi,:] = np.linalg.inv(XTX_sub)@ XTY_sub
    return W_sol, np.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) )


def greedy_CD(W, XTX, XTY, num_cin, num_sp, update_iter = 1):
            
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    Hess_inv = np.zeros((num_cin,ksize,ksize))
    for ih in range(num_cin):
        Hess_inv[ih,:,:] = np.linalg.inv(XTX[ih*ksize:(ih+1)*ksize,ih*ksize:(ih+1)*ksize] + 1e-12)
    
    W = np.linalg.inv(XTX)@ XTY
    num_prune = np.sum(np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12)
    prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
    
    for i1 in range(np.maximum(0,num_cin-num_sp-num_prune)):
        
        W_tmp = np.zeros_like(W)
        for i2 in range(num_cin):
            W_tmp[i2*ksize:(i2+1)*ksize,:] = XTX[i2*ksize:(i2+1)*ksize,i2*ksize:(i2+1)*ksize]@W[i2*ksize:(i2+1)*ksize,:]
        
        obj_cha = W * (-XTX@W + XTY + 0.5 * W_tmp)
        obj_cha = W * (0.5 * W_tmp)
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = np.sum(np.sum(obj_cha,axis=2),axis=1)
        
        idx = np.argsort(obj_sum + 1e20 * (prune_list) )
        
        W = W.reshape(num_cin, ksize, num_cout)
        W[idx[0],:,:] = 0
        W = W.reshape(totp, num_cout)
        prune_list[idx[0]] = True
        
        for i2 in range(update_iter):
            for i3 in range(num_cin):
                if prune_list[i3] == True:
                    continue
                W[i3*ksize:(i3+1)*ksize,:] = Hess_inv[i3,:,:] @ (XTY[i3*ksize:(i3+1)*ksize,:]-XTX[i3*ksize:(i3+1)*ksize,:]@W + XTX[i3*ksize:(i3+1)*ksize,i3*ksize:(i3+1)*ksize]@W[i3*ksize:(i3+1)*ksize,:])

    W_sol = np.zeros_like(W)
    nzi = np.nonzero(W[:,0])[0]
    XTX_sub = XTX[np.ix_(nzi,nzi)]
    XTY_sub = XTY[nzi,:]
    
    W_sol[nzi,:] = np.linalg.inv(XTX_sub)@ XTY_sub
    return W_sol, np.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) ) 

def greedy_thi(W, XTX, XTY, num_cin, num_sp):
            
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    W = np.linalg.inv(XTX)@ XTY
    num_prune = np.sum(np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12)
    prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
    
    for i1 in range(np.maximum(0,num_cin-num_sp-num_prune)):
        
        W_tmp = np.zeros_like(W)
        for i2 in range(num_cin):
            W_tmp[i2*ksize:(i2+1)*ksize,:] = XTX[i2*ksize:(i2+1)*ksize,i2*ksize:(i2+1)*ksize]@W[i2*ksize:(i2+1)*ksize,:]
        
        obj_cha = W * (-XTX@W + XTY + 0.5 * W_tmp)
        obj_cha = W * (0.5 * W_tmp)
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = np.sum(np.sum(obj_cha,axis=2),axis=1)
        
        idx = np.argsort(obj_sum + 1e20 * (prune_list) )
        
        W = W.reshape(num_cin, ksize, num_cout)
        W[idx[0],:,:] = 0
        W = W.reshape(totp, num_cout)
        prune_list[idx[0]] = True
        
    W_sol = np.zeros_like(W)
    nzi = np.nonzero(W[:,0])[0]
    XTX_sub = XTX[np.ix_(nzi,nzi)]
    XTY_sub = XTY[nzi,:]
    
    W_sol[nzi,:] = np.linalg.inv(XTX_sub)@ XTY_sub
    return W_sol, np.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) ) 



def greedy_CD_v2(W, XTX, XTY, num_cin, num_sp, update_iter = 1):
            
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    Hess_inv = np.zeros((num_cin,ksize,ksize))
    for ih in range(num_cin):
        Hess_inv[ih,:,:] = np.linalg.inv(XTX[ih*ksize:(ih+1)*ksize,ih*ksize:(ih+1)*ksize])
    
    XTX_inv = np.linalg.inv(XTX)
    W = XTX_inv@ XTY
    
    prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
    num_prune = np.sum(np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12)
    
    Hdiag = np.zeros((num_cin,ksize,ksize))
    for i1 in range(num_cin):
        Hdiag[i1,:,:] = np.copy(XTX_inv[i1*ksize:(i1+1)*ksize,i1*ksize:(i1+1)*ksize])
        
    for i1 in range(np.maximum(0,num_cin-num_sp-num_prune)):

        obj_mat = np.zeros_like(W)
        for i2 in range(num_cin):
            if prune_list[i2]:
                continue
            
            obj_mat[i2*ksize:(i2+1)*ksize,:] = np.linalg.inv(Hdiag[i2,:,:])@W[i2*ksize:(i2+1)*ksize,:] / 2

        obj_cha = W * obj_mat
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = np.sum(np.sum(obj_cha,axis=2),axis=1)
                                                             
        idx = np.argsort(obj_sum + 1e20 * (prune_list) )
        
        W = W.reshape(num_cin, ksize, num_cout)
        W[idx[0],:,:] = 0
        W = W.reshape(totp, num_cout)
        prune_list[idx[0]] = True
        
        Hinv = np.linalg.inv(Hdiag[idx[0],:,:])
        for i2 in range(num_cin):
            Hdiag[i2,:,:] -= XTX_inv[i2*ksize:(i2+1)*ksize,idx[0]*ksize:(idx[0]+1)*ksize] @ Hinv @ XTX_inv[idx[0]*ksize:(idx[0]+1)*ksize,i2*ksize:(i2+1)*ksize]
            
            S, U = np.linalg.eig(Hdiag[i2,:,:] - Hess_inv[i2,:,:])
            S = np.maximum(S,0)
            Hdiag[i2,:,:] = Hess_inv[i2,:,:] + U @ np.diag(S) @ U.T
        
        for i2 in range(update_iter):    
            for i3 in range(num_cin):
                if prune_list[i3] == True:
                    continue
                W[i3*ksize:(i3+1)*ksize,:] = Hess_inv[i3,:,:] @ (XTY[i3*ksize:(i3+1)*ksize,:]-XTX[i3*ksize:(i3+1)*ksize,:]@W + XTX[i3*ksize:(i3+1)*ksize,i3*ksize:(i3+1)*ksize]@W[i3*ksize:(i3+1)*ksize,:])

    W_sol = np.zeros_like(W)
    nzi = np.nonzero(W[:,0])[0]
    XTX_sub = XTX[np.ix_(nzi,nzi)]
    XTY_sub = XTY[nzi,:]
    
    W_sol[nzi,:] = np.linalg.inv(XTX_sub)@ XTY_sub
    return W_sol, np.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) ) 

def greedy_CD_hf(W, XTX, XTY, num_cin, num_sp, update_iter = 1):
            
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    Hess_inv = np.zeros((num_cin,ksize,ksize))
    for ih in range(num_cin):
        Hess_inv[ih,:,:] = np.linalg.inv(XTX[ih*ksize:(ih+1)*ksize,ih*ksize:(ih+1)*ksize] + 1e-12)
    
    num_prune = np.sum(np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12)
    prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
    
    
    for i2 in range(5*update_iter):
        for i3 in range(num_cin):
            if prune_list[i3] == True:
                continue
            W[i3*ksize:(i3+1)*ksize,:] = Hess_inv[i3,:,:] @ (XTY[i3*ksize:(i3+1)*ksize,:]-XTX[i3*ksize:(i3+1)*ksize,:]@W + XTX[i3*ksize:(i3+1)*ksize,i3*ksize:(i3+1)*ksize]@W[i3*ksize:(i3+1)*ksize,:])

    
    for i1 in range(np.maximum(0,num_cin-num_sp-num_prune)):
        
        W_tmp = np.zeros_like(W)
        for i2 in range(num_cin):
            W_tmp[i2*ksize:(i2+1)*ksize,:] = XTX[i2*ksize:(i2+1)*ksize,i2*ksize:(i2+1)*ksize]@W[i2*ksize:(i2+1)*ksize,:]
        
        obj_cha = W * (-XTX@W + XTY + 0.5 * W_tmp)
        obj_cha = W * (0.5 * W_tmp)
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = np.sum(np.sum(obj_cha,axis=2),axis=1)
        
        idx = np.argsort(obj_sum + 1e20 * (prune_list) )
        
        W = W.reshape(num_cin, ksize, num_cout)
        W[idx[0],:,:] = 0
        W = W.reshape(totp, num_cout)
        prune_list[idx[0]] = True
        
        for i2 in range(update_iter):
            for i3 in range(num_cin):
                if prune_list[i3] == True:
                    continue
                W[i3*ksize:(i3+1)*ksize,:] = Hess_inv[i3,:,:] @ (XTY[i3*ksize:(i3+1)*ksize,:]-XTX[i3*ksize:(i3+1)*ksize,:]@W + XTX[i3*ksize:(i3+1)*ksize,i3*ksize:(i3+1)*ksize]@W[i3*ksize:(i3+1)*ksize,:])

    for i2 in range(5*update_iter):
        for i3 in range(num_cin):
            if prune_list[i3] == True:
                continue
            W[i3*ksize:(i3+1)*ksize,:] = Hess_inv[i3,:,:] @ (XTY[i3*ksize:(i3+1)*ksize,:]-XTX[i3*ksize:(i3+1)*ksize,:]@W + XTX[i3*ksize:(i3+1)*ksize,i3*ksize:(i3+1)*ksize]@W[i3*ksize:(i3+1)*ksize,:])

    return W, np.sum( -W * XTY + (1/2) * W * (XTX@W) ) 

def forward_CD(W, XTX, XTY, num_cin, num_sp, update_iter = 1):
    
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    W = np.zeros_like(W)
    prune_list = np.zeros(num_cin)
    
    Hess_inv = np.zeros((num_cin,ksize,ksize))
    for ih in range(num_cin):
        Hess_inv[ih,:,:] = np.linalg.inv(XTX[ih*ksize:(ih+1)*ksize,ih*ksize:(ih+1)*ksize])
    prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
    for i1 in range(num_sp):
        
        G_tmp = np.zeros_like(W)
        W_tmp = np.zeros_like(W)
        for i2 in range(num_cin):
            G_tmp[i2*ksize:(i2+1)*ksize,:] = XTY[i2*ksize:(i2+1)*ksize,:] - XTX[i2*ksize:(i2+1)*ksize,:]@W + XTX[i2*ksize:(i2+1)*ksize,i2*ksize:(i2+1)*ksize]@W[i2*ksize:(i2+1)*ksize,:]
            W_tmp[i2*ksize:(i2+1)*ksize,:] = Hess_inv[i2,:,:] @ G_tmp[i2*ksize:(i2+1)*ksize,:]
        
        
        obj_cha = W_tmp * G_tmp
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = np.sum(np.sum(obj_cha,axis=2),axis=1)

        
        idx = np.argmin(-obj_sum + 1e20*(1-prune_list))   
        W[idx*ksize:(idx+1)*ksize,:] = np.copy(W_tmp[idx*ksize:(idx+1)*ksize,:])
        
        prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
        for i2 in range(update_iter):
            for i3 in range(num_cin):
                if prune_list[i3] == True:
                    continue
                W[i3*ksize:(i3+1)*ksize,:] = Hess_inv[i3,:,:] @ (XTY[i3*ksize:(i3+1)*ksize,:]-XTX[i3*ksize:(i3+1)*ksize,:]@W + XTX[i3*ksize:(i3+1)*ksize,i3*ksize:(i3+1)*ksize]@W[i3*ksize:(i3+1)*ksize,:])

    for i2 in range(5*update_iter):
        for i3 in range(num_cin):
            if prune_list[i3] == True:
                continue
            W[i3*ksize:(i3+1)*ksize,:] = Hess_inv[i3,:,:] @ (XTY[i3*ksize:(i3+1)*ksize,:]-XTX[i3*ksize:(i3+1)*ksize,:]@W + XTX[i3*ksize:(i3+1)*ksize,i3*ksize:(i3+1)*ksize]@W[i3*ksize:(i3+1)*ksize,:])

    return W, np.sum( -W * XTY + (1/2) * W * (XTX@W) ) 


def group_LASSO(W, XTX, XTY, lam1, lam2, W_2, num_cin, max_iter):
    
    alpha = 1 / np.linalg.norm(XTX, 2)
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    obj_list = np.zeros((max_iter,))
    
    def cal_obj(W):
        
        groupnorm = 0
        for ic in range(num_cin):
            groupnorm += np.linalg.norm(W[ic*ksize:(ic+1)*ksize,:])
        
        return np.sum( -W * XTY + (1/2) * W * (XTX@W) ) + lam2 * np.linalg.norm(W-W_2)**2 / 2 + lam1 * groupnorm
    
    def cal_grad(W):
        return XTX@W - XTY + lam2 * (W-W_2)
    
    def prox(W, alpha_k):
        
        norm_c = np.zeros((num_cin,))
        for ic in range(num_cin):
            norm_c[ic] = np.linalg.norm(W[ic*ksize:(ic+1)*ksize,:])
            
        norm_upd = np.where(norm_c > alpha_k, norm_c - alpha_k, 0)
        for ic in range(num_cin):
            
            if norm_c[ic] > 0:
                W[ic*ksize:(ic+1)*ksize,:] *= norm_upd[ic] / norm_c[ic]
                
        return W
    
    W_prev = np.copy(W)
    for ir in range(max_iter):
        
        W_Y = W + ir/(ir+3) * (W - W_prev)
        GY = cal_grad(W_Y)
        
        W_prev = np.copy(W)
        W = prox(W_Y - alpha * GY, alpha * lam1)
        
        obj_list[ir] = cal_obj(W)
        
    return W, obj_list

def refine_LASSO(W, XTX, XTY, num_cin, num_sp):
    
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    norm_c = np.zeros((num_cin,))
    for ic in range(num_cin):
        norm_c[ic] = np.linalg.norm(W[ic*ksize:(ic+1)*ksize,:])
    
    idx = np.argsort(-norm_c)
    norm_upd = np.zeros((num_cin,))
    norm_upd[idx[:num_sp]] = 1
    
    W_sol = np.zeros_like(W)
    for ic in range(num_cin):
        if norm_upd[ic] == 1:
            W_sol[ic*ksize:(ic+1)*ksize,:] = 1
    
    nzi = np.nonzero(W_sol[:,0])[0]
    XTX_sub = XTX[np.ix_(nzi,nzi)]
    XTY_sub = XTY[nzi,:]
    
    W_sol[nzi,:] = np.linalg.inv(XTX_sub)@ XTY_sub
    
    return W_sol, np.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) )

def local_search(W, XTX, XTY, num_cin, max_iter = 10, test_all = 5):
    
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    Hess_inv = np.zeros((num_cin,ksize,ksize))
    for ih in range(num_cin):
        Hess_inv[ih,:,:] = np.linalg.inv(XTX[ih*ksize:(ih+1)*ksize,ih*ksize:(ih+1)*ksize])
    
    flag_succ = 1
    for i_local in range(max_iter):
        
        if flag_succ == 1:
            flag_succ = 0
        else:
            break
            
        obj_cur = np.sum( -W * XTY + (1/2) * W * (XTX@W) ) 
        prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12

        obj_out = np.zeros((num_cin,))
        obj_in = np.zeros((num_cin,))
        obj_in2 = np.zeros((num_cin,))
        for i2 in range(num_cin):    
            if prune_list[i2]:
                continue
            obj_out[i2] =  np.sum(W[i2*ksize:(i2+1)*ksize,:] * (-XTX[i2*ksize:(i2+1)*ksize,:]@W + XTY[i2*ksize:(i2+1)*ksize,:] 
                                                         + 0.5 * XTX[i2*ksize:(i2+1)*ksize,i2*ksize:(i2+1)*ksize]@W[i2*ksize:(i2+1)*ksize,:]))
        
        idx_per2 = np.argsort(obj_out + 1e20*(np.abs(obj_out)<=1e-12))
        for idx2 in range(num_cin):
            i2 = idx_per2[idx2]
            if prune_list[i2] or idx2 >= test_all:
                continue

            for i3 in range(num_cin):
                if not prune_list[i3]:
                    continue
                w_grad =  (-XTY[i3*ksize:(i3+1)*ksize,:]+XTX[i3*ksize:(i3+1)*ksize,:]@W - XTX[i3*ksize:(i3+1)*ksize,i2*ksize:(i2+1)*ksize]@W[i2*ksize:(i2+1)*ksize,:])
                obj_in[i3] = 0.5 * np.sum(w_grad * (Hess_inv[i3,:,:] @ w_grad) )
        
            
            idx_per3 = np.argsort(-obj_in)
            for idx3 in range(num_cin):
                i3 = idx_per3[idx3]
                if not prune_list[i3] or idx3 >= test_all:
                    continue
            
                W_up = np.copy(W)
                W_up[i2*ksize:(i2+1)*ksize,:] = 0
                W_up[i3*ksize:(i3+1)*ksize,:] = 1
                nzi = np.nonzero(W_up[:,0])[0]
                XTX_sub = XTX[np.ix_(nzi,nzi)]
                XTY_sub = XTY[nzi,:]
                W_up[nzi,:] = np.linalg.inv(XTX_sub)@ XTY_sub
           
            
                obj_aft = np.sum( -W_up * XTY + (1/2) * W_up * (XTX@W_up) ) 
                obj_in2[i3] = obj_aft
                
                #print("Out: {}, in: {}, estimate obj is {}, {}, true obj is {}".format(i2,i3,obj_out[i2],obj_in[i3],obj_aft-obj_cur))
            
            
            if np.min(obj_in2 + 1e20*(np.abs(obj_in2)<=1e-12)) < obj_cur:
                
                i3 = np.argmin(obj_in2 + 1e20*(np.abs(obj_in2)<=1e-12))
                W[i2*ksize:(i2+1)*ksize,:] = 0
                W[i3*ksize:(i3+1)*ksize,:] = 1
                nzi = np.nonzero(W[:,0])[0]
                XTX_sub = XTX[np.ix_(nzi,nzi)]
                XTY_sub = XTY[nzi,:]
                W[nzi,:] = np.linalg.inv(XTX_sub)@ XTY_sub
                
                obj_cur = np.min(obj_in2)
                prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
                print("Finish iter {}, obj is {}".format(i_local,obj_cur))
                flag_succ = 1
                break

    W_sol = np.copy(W)
    
    return W_sol, np.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) ) 

def backward_selection(W, XTX, XTY, num_cin, num_sp):
    
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
    prune_list2 = np.where(np.abs(np.sum(W,axis=1)) >= 1e-12)[0]
    num_prune = np.sum(prune_list)
    
    XTX_inv = np.zeros_like(XTX)
    XTX_inv[prune_list2[:, np.newaxis], prune_list2] = np.linalg.inv(XTX[prune_list2[:, np.newaxis], prune_list2])
    
    W = XTX_inv @ XTY
    
    for i1 in range(np.maximum(0,num_cin-num_sp-num_prune)):

        
        
        obj_mat = np.zeros_like(W)
        for i2 in range(num_cin):
            if prune_list[i2]:
                continue
            obj_mat[i2*ksize:(i2+1)*ksize,:] = np.linalg.inv(XTX_inv[i2*ksize:(i2+1)*ksize,i2*ksize:(i2+1)*ksize])@W[i2*ksize:(i2+1)*ksize,:] / 2
        
        obj_cha = W * obj_mat
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = np.sum(np.sum(obj_cha,axis=2),axis=1)

        idx = np.argsort(obj_sum + 1e20*(prune_list) )
       
        W -= XTX_inv[:,idx[0]*ksize:(idx[0]+1)*ksize] @ np.linalg.inv(XTX_inv[idx[0]*ksize:(idx[0]+1)*ksize,idx[0]*ksize:(idx[0]+1)*ksize]) @ W[idx[0]*ksize:(idx[0]+1)*ksize,:]
        W[idx[0]*ksize:(idx[0]+1)*ksize,:] = 0
        
        prune_list[idx[0]] = True
        prune_list2 = np.where(np.abs(np.sum(W,axis=1)) >= 1e-12)[0]
        
        XTX_inv[prune_list2[:, np.newaxis], prune_list2] += (XTX_inv[prune_list2[:, np.newaxis], prune_list2]@XTX[prune_list2,idx[0]*ksize:(idx[0]+1)*ksize])@np.linalg.inv(np.eye(ksize)-XTX_inv[idx[0]*ksize:(idx[0]+1)*ksize,prune_list2]@XTX[prune_list2,idx[0]*ksize:(idx[0]+1)*ksize])@XTX_inv[idx[0]*ksize:(idx[0]+1)*ksize,prune_list2]
        
        
        XTX_inv[idx[0]*ksize:(idx[0]+1)*ksize,:] = 0
        XTX_inv[:,idx[0]*ksize:(idx[0]+1)*ksize] = 0
        
      
    W_sol = np.zeros_like(W)
    nzi = np.nonzero(W[:,0])[0]
    XTX_sub = XTX[np.ix_(nzi,nzi)]
    XTY_sub = XTY[nzi,:]
    
    W_sol[nzi,:] = np.linalg.inv(XTX_sub)@ XTY_sub
    return W_sol, np.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) ) 


def backward_lowrank(W, XTX, XTY, num_cin, num_sp, low_ratio):
    
    
    def get_decomp(H, num_cin, lowr):
    
        totp = H.shape[0]
        ksize = int(totp / num_cin)
        U, S, V = randomized_svd(H, lowr,random_state=0,power_iteration_normalizer="QR") 
        H2 = H - U * S @ U.T
        D = np.zeros((totp, ksize))
        for i1 in range(num_cin):
            D[i1*ksize:(i1+1)*ksize,:] = H2[i1*ksize:(i1+1)*ksize,i1*ksize:(i1+1)*ksize]
        Hnew = np.zeros_like(H)
        for i1 in range(num_cin):
            Hnew[i1*ksize:(i1+1)*ksize,i1*ksize:(i1+1)*ksize] = D[i1*ksize:(i1+1)*ksize,:]
    
        Hnew += U * S @ U.T
        diff = np.linalg.norm(Hnew - H, 2) / np.linalg.norm(H, 2)
    
        return D, U * (S**(1/2)), diff
    
    
    D, L, _ = get_decomp(np.copy(XTX), num_cin, int(XTX.shape[0] * low_ratio))
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    lowr = L.shape[1]
    prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
    prune_list2 = np.where(np.abs(np.sum(W,axis=1)) >= 1e-12)[0]
    num_prune = np.sum(prune_list)
    
    Dinv = np.zeros_like(D)
    for i1 in range(num_cin):
        Dinv[i1*ksize:(i1+1)*ksize,:] = np.linalg.inv(D[i1*ksize:(i1+1)*ksize,:])  
    
    def diagproduct(Diag, MAT):
        
        MAT2 = np.zeros_like(MAT)
        for i1 in range(num_cin):
            MAT2[i1*ksize:(i1+1)*ksize,:] = Diag[i1*ksize:(i1+1)*ksize,:] @ MAT[i1*ksize:(i1+1)*ksize,:]
            
        return MAT2
    
    Cinv = np.linalg.inv(np.eye(lowr) + L.T @ diagproduct(Dinv, L))
    Linv = diagproduct(Dinv, L) 
    Hess_diag = np.zeros((num_cin, ksize, ksize))
    for i1 in range(num_cin):
        Hess_diag[i1,:,:] = Dinv[i1*ksize:(i1+1)*ksize,:] - Linv[i1*ksize:(i1+1)*ksize,:] @ Cinv @ Linv[i1*ksize:(i1+1)*ksize,:].T

    W = diagproduct(Dinv, XTY) - (Linv @ (Cinv @ (Linv.T @ XTY)))
    for i1 in range(np.maximum(0,num_cin-num_sp-num_prune)):
        
        obj_mat = np.zeros_like(W)
        for i2 in range(num_cin):
            if prune_list[i2]:
                continue
            obj_mat[i2*ksize:(i2+1)*ksize,:] = np.linalg.inv(Hess_diag[i2,:,:]) @ W[i2*ksize:(i2+1)*ksize,:] / 2
        
        obj_cha = W * obj_mat
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = np.sum(np.sum(obj_cha,axis=2),axis=1)
        idx = np.argsort(obj_sum + 1e20*(prune_list) )
       
        Hinv_band = - Linv @ (Cinv @ Linv[idx[0]*ksize:(idx[0]+1)*ksize,:].T) 
        Hinv_band[idx[0]*ksize:(idx[0]+1)*ksize,:] += Dinv[idx[0]*ksize:(idx[0]+1)*ksize,:] 
        
        
        
        W -= Hinv_band @ (np.linalg.inv(Hess_diag[idx[0],:,:]) @ W[idx[0]*ksize:(idx[0]+1)*ksize,:])

        W[idx[0]*ksize:(idx[0]+1)*ksize,:] = 0
        prune_list[idx[0]] = True
        prune_list2 = np.where(np.abs(np.sum(W,axis=1)) >= 1e-12)[0]
        
        CL = Cinv @ Linv[idx[0]*ksize:(idx[0]+1)*ksize,:].T
        Einv = np.linalg.inv(Dinv[idx[0]*ksize:(idx[0]+1)*ksize,:] - Linv[idx[0]*ksize:(idx[0]+1)*ksize,:] @ CL)
        
        for i3 in range(num_cin):
            if prune_list[i3]:
                continue
                
            LCL = Linv[i3*ksize:(i3+1)*ksize,:] @ CL
            Hess_diag[i3,:,:] -= LCL @ Einv @ LCL.T
        
        Cinv = Cinv + CL @ Einv @ CL.T
        
       
        Linv[idx[0]*ksize:(idx[0]+1)*ksize,:] = 0
    
    return weight_update(W, XTX, XTY)

def backward_lowrank2(W, XTX, XTY, num_cin, num_sp, low_ratio):
    
    
    def get_decomp(H, num_cin, lowr):
    
        totp = H.shape[0]
        ksize = int(totp / num_cin)

        U, S, V = randomized_svd(H, lowr,random_state=0,power_iteration_normalizer="QR") 

        H2 = H - U * S @ U.T
        D = np.zeros((totp, ksize))
        for i1 in range(num_cin):
            D[i1*ksize:(i1+1)*ksize,:] = H2[i1*ksize:(i1+1)*ksize,i1*ksize:(i1+1)*ksize]
        
        Hnew = np.zeros_like(H)
        for i1 in range(num_cin):
            Hnew[i1*ksize:(i1+1)*ksize,i1*ksize:(i1+1)*ksize] = D[i1*ksize:(i1+1)*ksize,:]
    
        Hnew += U * S @ U.T
        diff = np.linalg.norm(Hnew - H, 2) / np.linalg.norm(H, 2)
    
        return D, U * (S**(1/2)), diff
    
    
    D, L, _ = get_decomp(np.copy(XTX), num_cin, int(XTX.shape[0] * low_ratio))
    
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    lowr = L.shape[1]
    
    prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
    prune_list2 = np.where(np.abs(np.sum(W,axis=1)) >= 1e-12)[0]
    num_prune = np.sum(prune_list)
    
    Dinv = np.zeros_like(D)
    for i1 in range(num_cin):
        Dinv[i1*ksize:(i1+1)*ksize,:] = np.linalg.inv(D[i1*ksize:(i1+1)*ksize,:])
        
    
    def diagproduct(Diag, MAT):
        
        MAT2 = np.zeros_like(MAT)
        for i1 in range(num_cin):
            MAT2[i1*ksize:(i1+1)*ksize,:] = Diag[i1*ksize:(i1+1)*ksize,:] @ MAT[i1*ksize:(i1+1)*ksize,:]
            
        return MAT2
    
    Cinv = np.linalg.inv(np.eye(lowr) + L.T @ diagproduct(Dinv, L))
    Linv = diagproduct(Dinv, L)
    
    Hess_diag = np.zeros((num_cin, ksize, ksize))
    for i1 in range(num_cin):
        Hess_diag[i1,:,:] = Dinv[i1*ksize:(i1+1)*ksize,:] - Linv[i1*ksize:(i1+1)*ksize,:] @ Cinv @ Linv[i1*ksize:(i1+1)*ksize,:].T

    W = np.linalg.inv(XTX) @ XTY
    
    for i1 in range(np.maximum(0,num_cin-num_sp-num_prune)):
        
        obj_mat = np.zeros_like(W)
        for i2 in range(num_cin):
            if prune_list[i2]:
                continue
            obj_mat[i2*ksize:(i2+1)*ksize,:] = np.linalg.inv(Hess_diag[i2,:,:]) @ W[i2*ksize:(i2+1)*ksize,:] / 2
        
        obj_cha = W * obj_mat
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = np.sum(np.sum(obj_cha,axis=2),axis=1)
        idx = np.argsort(obj_sum + 1e20*(prune_list) )
       
        Hinv_band = - Linv @ (Cinv @ Linv[idx[0]*ksize:(idx[0]+1)*ksize,:].T) 
        Hinv_band[idx[0]*ksize:(idx[0]+1)*ksize,:] += Dinv[idx[0]*ksize:(idx[0]+1)*ksize,:] 
        
        
        
        W -= Hinv_band @ (np.linalg.inv(Hess_diag[idx[0],:,:]) @ W[idx[0]*ksize:(idx[0]+1)*ksize,:])

        W[idx[0]*ksize:(idx[0]+1)*ksize,:] = 0
        prune_list[idx[0]] = True
        prune_list2 = np.where(np.abs(np.sum(W,axis=1)) >= 1e-12)[0]
        
        CL = Cinv @ Linv[idx[0]*ksize:(idx[0]+1)*ksize,:].T
        Einv = np.linalg.inv(Dinv[idx[0]*ksize:(idx[0]+1)*ksize,:] - Linv[idx[0]*ksize:(idx[0]+1)*ksize,:] @ CL)
        
        for i3 in range(num_cin):
            if prune_list[i3]:
                continue
                
            LCL = Linv[i3*ksize:(i3+1)*ksize,:] @ CL
            Hess_diag[i3,:,:] -= LCL @ Einv @ LCL.T
        
        Cinv = Cinv + CL @ Einv @ CL.T
        
       
        Linv[idx[0]*ksize:(idx[0]+1)*ksize,:] = 0
    
    return weight_update(W, XTX, XTY)

def backward_lowrank3(W, XTX, XTY, num_cin, num_sp, low_ratio):
    
    
    def get_decomp(H, num_cin, lowr):
    
        totp = H.shape[0]
        ksize = int(totp / num_cin)

        U, S, V = randomized_svd(H, lowr,random_state=0,power_iteration_normalizer="QR") 

        H2 = H - U * S @ U.T
        D = np.zeros((totp, ksize))
        for i1 in range(num_cin):
            D[i1*ksize:(i1+1)*ksize,:] = H2[i1*ksize:(i1+1)*ksize,i1*ksize:(i1+1)*ksize]
        
        Hnew = np.zeros_like(H)
        for i1 in range(num_cin):
            Hnew[i1*ksize:(i1+1)*ksize,i1*ksize:(i1+1)*ksize] = D[i1*ksize:(i1+1)*ksize,:]
    
        Hnew += U * S @ U.T
        diff = np.linalg.norm(Hnew - H, 2) / np.linalg.norm(H, 2)
    
        return D, U * (S**(1/2)), diff
    
    XTX_inv = np.linalg.inv(XTX)
    D, L, _ = get_decomp(XTX_inv, num_cin, int(XTX.shape[0] * low_ratio))
    
    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    lowr = L.shape[1]
    
    prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
    prune_list2 = np.where(np.abs(np.sum(W,axis=1)) >= 1e-12)[0]
    num_prune = np.sum(prune_list)
        
    
    def diagproduct(Diag, MAT):
        
        MAT2 = np.zeros_like(MAT)
        for i1 in range(num_cin):
            MAT2[i1*ksize:(i1+1)*ksize,:] = Diag[i1*ksize:(i1+1)*ksize,:] @ MAT[i1*ksize:(i1+1)*ksize,:]
            
        return MAT2
    
    C = np.eye(lowr)
    
    Hess_diag = np.zeros((num_cin, ksize, ksize))
    for i1 in range(num_cin):
        Hess_diag[i1,:,:] = D[i1*ksize:(i1+1)*ksize,:] + L[i1*ksize:(i1+1)*ksize,:] @ L[i1*ksize:(i1+1)*ksize,:].T

    #W = diagproduct(D, XTY) + (L @ (C @ (L.T @ XTY)))
    W = XTX_inv @ XTY
    for i1 in range(np.maximum(0,num_cin-num_sp-num_prune)):
        
        obj_mat = np.zeros_like(W)
        for i2 in range(num_cin):
            if prune_list[i2]:
                continue
            obj_mat[i2*ksize:(i2+1)*ksize,:] = np.linalg.inv(Hess_diag[i2,:,:]) @ W[i2*ksize:(i2+1)*ksize,:] / 2
        
        obj_cha = W * obj_mat
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = np.sum(np.sum(obj_cha,axis=2),axis=1)
        idx = np.argsort(obj_sum + 1e20*(prune_list) )
       
        Hinv_band = L @ (C @ L[idx[0]*ksize:(idx[0]+1)*ksize,:].T) 
        Hinv_band[idx[0]*ksize:(idx[0]+1)*ksize,:] += D[idx[0]*ksize:(idx[0]+1)*ksize,:] 
        
        
        
        W -= Hinv_band @ (np.linalg.inv(Hess_diag[idx[0],:,:]) @ W[idx[0]*ksize:(idx[0]+1)*ksize,:])

        W[idx[0]*ksize:(idx[0]+1)*ksize,:] = 0
        prune_list[idx[0]] = True
        prune_list2 = np.where(np.abs(np.sum(W,axis=1)) >= 1e-12)[0]
        
        CL = C @ L[idx[0]*ksize:(idx[0]+1)*ksize,:].T
        Einv = np.linalg.inv(D[idx[0]*ksize:(idx[0]+1)*ksize,:] + L[idx[0]*ksize:(idx[0]+1)*ksize,:] @ CL)
        
        for i3 in range(num_cin):
            if prune_list[i3]:
                continue
                
            LCL = L[i3*ksize:(i3+1)*ksize,:] @ CL
            Hess_diag[i3,:,:] -= LCL @ Einv @ LCL.T
        
        C = C - CL @ Einv @ CL.T
        
       
        L[idx[0]*ksize:(idx[0]+1)*ksize,:] = 0
        
        
    
    return weight_update(W, XTX, XTY)


def greedy_CD_v3(W, XTX, XTY, num_cin, num_sp, update_iter = 1):

    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    XTX_inv = np.linalg.inv(XTX)
    
    W = XTX_inv@ XTY
    
    num_prune = np.sum(np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12)
    prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
    
    
    
    if int(num_cin-num_sp-num_prune) <= 0:
        upd_it = 0
    else:
        upd_it = int((num_cin-num_sp-num_prune) / update_iter)
        if upd_it == 0:
            upd_it = 1
        quo, rem = divmod(int(num_cin-num_sp-num_prune), int(upd_it))
        update_ten = np.full((upd_it,), quo, dtype=int)
        update_ten[:rem] += 1
    
    for i1 in range(upd_it):
        
        obj_mat = np.zeros_like(W)
        if ksize > 1:
            for i2 in range(num_cin):
                if prune_list[i2]:
                    continue
                obj_mat[i2*ksize:(i2+1)*ksize,:] = np.linalg.inv(XTX_inv[i2*ksize:(i2+1)*ksize,i2*ksize:(i2+1)*ksize])@W[i2*ksize:(i2+1)*ksize,:] / 2
        else:
            obj_mat = (1 / (prune_list + np.diag(XTX_inv)))[:,None] * W / 2


        obj_cha = W * obj_mat
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = np.sum(np.sum(obj_cha,axis=2),axis=1)
        
        idx = np.argsort(obj_sum + 1e20 * (prune_list) )
        
        

        upd_idx = np.concatenate([np.arange(idx[i]*ksize, (idx[i]+1)*ksize) for i in range(update_ten[i1])])

        
        Xinv_tmp = np.linalg.inv(XTX_inv[upd_idx[:,None],upd_idx])
        
        W -= XTX_inv[:,upd_idx] @ Xinv_tmp @ W[upd_idx,:]
        W = W.reshape(num_cin, ksize, num_cout)
        W[idx[:update_ten[i1]],:,:] = 0
        W = W.reshape(totp, num_cout)
    
        XTX_inv -= XTX_inv[:,upd_idx] @ Xinv_tmp @ XTX_inv[upd_idx,:]
        XTX_inv[upd_idx,:] = 0
        XTX_inv[:,upd_idx] = 0
    
        prune_list[idx[:update_ten[i1]]] = True

         
    W_sol = np.zeros_like(W)
    nzi = np.nonzero(W[:,0])[0]
    W_sol[nzi,:] = np.linalg.inv(XTX[nzi[:,None],nzi])@ XTY[nzi,:]
    
    return W_sol, np.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) ) 


def greedy_CD_v4(W, XTX, XTY, num_cin, num_sp, update_iter = 1):

    totp, num_cout = W.shape
    ksize = int(totp / num_cin)
    
    XTX_inv = np.linalg.inv(XTX)
    
    W = XTX_inv@ XTY
    
    num_prune = np.sum(np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12)
    prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
    
    
    
    if int(num_cin-num_sp-num_prune) <= 0:
        upd_it = 0
    else:
        upd_it = int((num_cin-num_sp-num_prune) / update_iter)
        if upd_it == 0:
            upd_it = 1
        quo, rem = divmod(int(num_cin-num_sp-num_prune), int(upd_it))
        update_ten = np.full((upd_it,), quo, dtype=int)
        update_ten[:rem] += 1
    
    for i1 in range(upd_it):
        
        W_tmp = np.zeros_like(W)
        if ksize > 1:
            for i2 in range(num_cin):
                W_tmp[i2*ksize:(i2+1)*ksize,:] = XTX[i2*ksize:(i2+1)*ksize,i2*ksize:(i2+1)*ksize]@W[i2*ksize:(i2+1)*ksize,:]
        else:
            W_tmp = np.diag(XTX)[:,None] * W 


        obj_cha = W * (-XTX@W + XTY + 0.5 * W_tmp)
        obj_cha = W * (0.5 * W_tmp)
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = np.sum(np.sum(obj_cha,axis=2),axis=1)
        
        idx = np.argsort(obj_sum + 1e20 * (prune_list) )
        
        
        

        upd_idx = np.concatenate([np.arange(idx[i]*ksize, (idx[i]+1)*ksize) for i in range(update_ten[i1])])

        
        Xinv_tmp = np.linalg.inv(XTX_inv[upd_idx[:,None],upd_idx])
        
        W -= XTX_inv[:,upd_idx] @ Xinv_tmp @ W[upd_idx,:]
        W = W.reshape(num_cin, ksize, num_cout)
        W[idx[:update_ten[i1]],:,:] = 0
        W = W.reshape(totp, num_cout)
    
        XTX_inv -= XTX_inv[:,upd_idx] @ Xinv_tmp @ XTX_inv[upd_idx,:]
        XTX_inv[upd_idx,:] = 0
        XTX_inv[:,upd_idx] = 0
    
        prune_list[idx[:update_ten[i1]]] = True

         
    W_sol = np.zeros_like(W)
    nzi = np.nonzero(W[:,0])[0]
    W_sol[nzi,:] = np.linalg.inv(XTX[nzi[:,None],nzi])@ XTY[nzi,:]
    
    return W_sol, np.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) ) 

def local_search_v3(W, XTX, XTY, num_cin, max_iter = 10, switch_ratio = 0.005, switch_lb = 0.001):
    
   
    totp, num_cout = W.shape
    
    num_swap = int(np.ceil(num_cin * switch_ratio))
    lb_swap = int(np.ceil(num_cin * switch_lb))
    ksize = int(totp / num_cin)
    
    prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
       
    best_prune = np.copy(prune_list)
    supp_idx = np.concatenate([np.arange(i*ksize, (i+1)*ksize) for i in range(num_cin) if not prune_list[i]])
    
    XTX_inv = np.zeros_like(XTX)
    XTX_inv[supp_idx[:,None],supp_idx] = np.linalg.inv(XTX[supp_idx[:,None],supp_idx])

    
    obj_cur = np.sum( -W * XTY + (1/2) * W* (XTX@W) ) 
    
    for i_local in range(max_iter):

        obj_mat = np.zeros_like(W)
        if ksize > 1:
            for i2 in range(num_cin):
                if prune_list[i2]:
                    continue
                obj_mat[i2*ksize:(i2+1)*ksize,:] = np.linalg.inv(XTX_inv[i2*ksize:(i2+1)*ksize,i2*ksize:(i2+1)*ksize])@W[i2*ksize:(i2+1)*ksize,:] / 2
        else:
            obj_mat = (1 / (prune_list + np.diag(XTX_inv)))[:,None] * W / 2


        obj_cha = W * obj_mat
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = np.sum(np.sum(obj_cha,axis=2),axis=1)
        
        idx = np.argsort(obj_sum + 1e20 * (prune_list) )
        
        upd_idx = np.concatenate([np.arange(idx[i]*ksize, (idx[i]+1)*ksize) for i in range(num_swap)])

        
        Xinv_tmp = np.linalg.inv(XTX_inv[upd_idx[:,None],upd_idx])
        W -= XTX_inv[:,upd_idx] @ Xinv_tmp @ W[upd_idx,:]
        W = W.reshape(num_cin, ksize, num_cout)
        W[idx[:num_swap],:,:] = 0
        W = W.reshape(totp, num_cout)
    
        XTX_inv -= XTX_inv[:,upd_idx] @ Xinv_tmp @ XTX_inv[upd_idx,:]
        XTX_inv[upd_idx,:] = 0
        XTX_inv[:,upd_idx] = 0
    
        prune_list[idx[:num_swap]] = True
        
        
            
        obj_in = np.zeros((num_cin,))

            
        supp_idx = np.concatenate([np.arange(i*ksize, (i+1)*ksize) for i in range(num_cin) if not prune_list[i]])
        H_inv = XTX_inv[supp_idx[:,None],supp_idx]
        H_invG = H_inv @ XTY[supp_idx,:]
            
            

        if ksize >= 2:
            for i3 in range(num_cin):
                if not prune_list[i3]:
                    continue
                        
                    
                b_ori = XTX[supp_idx,i3*ksize:(i3+1)*ksize]
                C_inv = np.linalg.inv(XTX[i3*ksize:(i3+1)*ksize,i3*ksize:(i3+1)*ksize] - b_ori.T @ H_inv @ b_ori)
                    
                gt = XTY[i3*ksize:(i3+1)*ksize,:] - b_ori.T @ H_invG
                obj_in[i3] = np.sum(gt * (C_inv @ gt)) / 2
                    
                #W1 = torch.clone(W)
                #W1[i2*ksize:(i2+1)*ksize,:] = 0
                #nzi = torch.nonzero(W1[:,0], as_tuple=True)[0]
                #XTX_sub = XTX[nzi[:,None],nzi]
                #XTY_sub = XTY[nzi,:]
                #W1[nzi,:] = torch.linalg.inv(XTX_sub)@ XTY_sub
                #obj1 = torch.sum( -W1 * XTY + (1/2) * W1 * (XTX @ W1)) 
                    
                #W2 = torch.clone(W)
                #W2[i2*ksize:(i2+1)*ksize,:] = 0
                #W2[i3*ksize:(i3+1)*ksize,:] = 1
                #nzi = torch.nonzero(W2[:,0], as_tuple=True)[0]
                #XTX_sub = XTX[nzi[:,None],nzi]
                #XTY_sub = XTY[nzi,:]
                #W2[nzi,:] = torch.linalg.inv(XTX_sub)@ XTY_sub
                #obj2 = torch.sum( -W2 * XTY + (1/2) * W2 * (XTX @ W2)) 
                    
                #print("Out: {}, in: {}, true obj is ori: {}, out: {}".format(i2,i3,obj1-obj_cur,obj1-obj2))
                #print("Out: {}, in: {}, estimate obj is out: {}, in: {}".format(i2,i3,obj_sum[i2],obj_in[i3]))
                         
 
        else:
            C_list = 1 / (np.diag(XTX) -  np.sum(XTX[supp_idx,:] * (H_inv @ XTX[supp_idx,:] ), axis = 0) + (~prune_list)*1e-8 ) 
            gt = XTY - XTX[:,supp_idx] @ H_invG
            obj_in = np.sum(gt**2, axis=1) * C_list / 2
            
        
        idx2 = np.argsort(-obj_in + 1e20*(~prune_list))
        
        prune_list[idx2[:num_swap]] = False   
        supp_idx = np.concatenate([np.arange(i*ksize, (i+1)*ksize) for i in range(num_cin) if not prune_list[i]])
    
        XTX_inv = np.zeros_like(XTX)
        XTX_inv[supp_idx[:,None],supp_idx] = np.linalg.inv(XTX[supp_idx[:,None],supp_idx])
                
        W = np.zeros_like(W)
        W = XTX_inv @ XTY
                
        obj_new = np.sum( -W * XTY + (1/2) * W * (XTX @ W))         
                
        print("Finish iter {}, old obj is {}, new is {}, numswap is {}".format(i_local, obj_cur, obj_new, num_swap))
        
        if obj_new < obj_cur:
            
            best_prune = np.copy(prune_list)
            obj_cur = obj_new
        else:
            if num_swap <= lb_swap:
                break
            else:
                num_swap = int(np.maximum(num_swap/2, lb_swap))
                
                prune_list = np.copy(best_prune)
                supp_idx = np.concatenate([np.arange(i*ksize, (i+1)*ksize) for i in range(num_cin) if not prune_list[i]])
    
                XTX_inv = np.zeros_like(XTX)
                XTX_inv[supp_idx[:,None],supp_idx] = np.linalg.inv(XTX[supp_idx[:,None],supp_idx])
                
                W = np.zeros_like(W)
                W = XTX_inv @ XTY

    
    supp_idx = np.concatenate([np.arange(i*ksize, (i+1)*ksize) for i in range(num_cin) if not best_prune[i]])
    
    W = np.zeros_like(W)
    W[supp_idx,:] = np.linalg.inv(XTX[supp_idx[:,None],supp_idx]) @ XTY[supp_idx,:]
    
    return W, np.sum( -W * XTY + (1/2) * W * (XTX@W) )

### Unstructured pruning


def weight_update_unstr(W, XTX, XTY):
    
    p, m = W.shape
    W_sol = np.zeros_like(W)
    for k in range(m):
        
        nzi = np.nonzero(W[:,k])[0]
        if len(nzi) == 0:
            continue
        XTX_sub = XTX[np.ix_(nzi,nzi)]
        XTY_sub = XTY[nzi,k]
        W_sol[nzi,k] = np.linalg.solve(XTX_sub, XTY_sub)
        
    return W_sol, np.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) ) 

def weight_selection_unstr(W, XTX, XTY, k_spar):
    
    p, m = W.shape
    
    W_vec = np.copy(W).reshape(-1)
    W_sort = np.argsort(-np.abs(W_vec))
    W_vec[W_sort[k_spar:]] = 0
    
    W_sol = W_vec.reshape(p,m)
    for k in range(m):
        
        nzi = np.nonzero(W_sol[:,k])[0]
        if len(nzi) == 0:
            continue
        XTX_sub = XTX[np.ix_(nzi,nzi)]
        XTY_sub = XTY[nzi,k]
        W_sol[nzi,k] = np.linalg.solve(XTX_sub, XTY_sub)
        
    return W_sol, np.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) ) 


def greedy_CD_unstr(W, XTX, XTY, k_spar, update_iter = 1):
    
    if update_iter >= 1:
        i_repeat = 1
        update_iter = int(update_iter)
    else:
        i_repeat = int(1/update_iter)
        update_iter = 1
    
    totp, num_cout = W.shape
    
    W_vec = np.copy(W).reshape(-1)
    W_sort = np.argsort(-np.abs(W_vec))
    W_vec[W_sort[k_spar:]] = 0
    
    W_sol = W_vec.reshape(totp, num_cout)
    
    XTX_inv = np.linalg.inv(XTX)
    for i_c in range(num_cout):
        
        mask = np.where(W_sol[:,i_c] != 0)[0] 
        k_sp = len(mask)
         
        w_prune = XTX_inv @ XTY[:,i_c]
        mask_idx = np.where(w_prune != 0, 0, np.inf)
        k_init = np.count_nonzero(w_prune)
        
        obj_sum = w_prune * (-XTX@w_prune + XTY[:,i_c] + 0.5 * np.diag(XTX) * w_prune)
        
        for i_t in range(k_init-k_sp):
            
            prune_idx = np.argmin(obj_sum + mask_idx)
            w_tmp = w_prune[prune_idx]
            w_prune[prune_idx] = 0
            mask_idx[prune_idx] = np.inf
            if (i_t+1)% update_iter == 0:
                
                for _ in range(i_repeat):
                    for i_cd in range(totp):   
                        if mask_idx[i_cd] < 1:
                            w_prune[i_cd] = (XTY[i_cd,i_c]- XTX[i_cd,:]@w_prune + XTX[i_cd,i_cd] * w_prune[i_cd]) / XTX[i_cd,i_cd]
                
                obj_sum = w_prune * (-XTX@w_prune + XTY[:,i_c] + 0.5 * np.diag(XTX) * w_prune)
            
            else:
                obj_sum += w_tmp * (w_prune * XTX[:,prune_idx])
                obj_sum[prune_idx] = 0
                

        W_sol[:,i_c] = np.copy(w_prune)
        

    return weight_update_unstr(W_sol, XTX, XTY)

def greedy_CD2_unstr(W, XTX, XTY, k_spar, update_iter = 1):

    if update_iter >= 1:
        i_repeat = 1
        update_iter = int(update_iter)
    else:
        i_repeat = int(1/update_iter)
        update_iter = 1
        
    totp, num_cout = W.shape
    
    W_vec = np.copy(W).reshape(-1)
    W_sort = np.argsort(-np.abs(W_vec))
    W_vec[W_sort[k_spar:]] = 0
    
    W_sol = W_vec.reshape(totp, num_cout)
    
    for i_c in range(num_cout):
        mask = np.where(W_sol[:,i_c] != 0)[0] 
        k_sp = len(mask)
        
        XTX_inv = np.linalg.inv(XTX)
        
        w_prune = XTX_inv @ XTY[:,i_c]
        mask_idx = np.where(w_prune != 0, 0, np.inf)
        k_init = np.count_nonzero(w_prune)
        Hdiag = np.copy(np.diag(XTX_inv))
        
        for i_t in range(k_init-k_sp):
            
            prune_idx = np.argmin(w_prune **2 / Hdiag + mask_idx)
            w_prune[prune_idx] = 0
            mask_idx[prune_idx] = np.inf

            
            Hdiag = Hdiag - XTX_inv[prune_idx,:] **2 / Hdiag[prune_idx]
            Hdiag = np.maximum(Hdiag, 1/np.diag(XTX))
            
            
            if (i_t+1)% update_iter == 0:
                
                for _ in range(i_repeat):
                    for i_cd in range(totp):   
                        if mask_idx[i_cd] < 1:
                            w_prune[i_cd] = (XTY[i_cd,i_c]-XTX[i_cd,:]@w_prune + XTX[i_cd,i_cd] * w_prune[i_cd]) / XTX[i_cd,i_cd]

        W_sol[:,i_c] = np.copy(w_prune)
        

    return weight_update_unstr(W_sol, XTX, XTY)

def backward_selection_unstr(W, XTX, XTY, k_spar):

    totp, num_cout = W.shape
    
    W_vec = np.copy(W).reshape(-1)
    W_sort = np.argsort(-np.abs(W_vec))
    W_vec[W_sort[k_spar:]] = 0
    
    W_sol = W_vec.reshape(totp, num_cout)
    
    for i_c in range(num_cout):
        
        mask = np.where(W_sol[:,i_c] != 0)[0] 
        k_sp = len(mask)
        
        XTX_inv = np.linalg.inv(XTX)
        
        w_prune = XTX_inv @ XTY[:,i_c]
        mask_idx = np.where(w_prune != 0, 0, np.inf)
        k_init = np.count_nonzero(w_prune)
        
        for i_t in range(k_init-k_sp):
            
            prune_idx = np.argmin( (w_prune)**2 / (np.diag(XTX_inv) + mask_idx) + mask_idx)
            
            mask_idx[prune_idx] = np.inf
            w_prune -= (w_prune[prune_idx] /  XTX_inv[prune_idx,prune_idx]) * XTX_inv[:,prune_idx] 
            w_prune[prune_idx] = 0
            
        
            XTX_inv -= XTX_inv[:,prune_idx:prune_idx+1] @ XTX_inv[prune_idx:prune_idx+1,:] / XTX_inv[prune_idx,prune_idx]
        
            XTX_inv[:,prune_idx] = 0
            XTX_inv[prune_idx,:] = 0
            
                          
        W_sol[:,i_c] = np.copy(w_prune)
        
        
    
    return W_sol, np.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) ) 


def backward_lowrank_unstr(W, XTX, XTY, k_spar, low_ratio, update_iter):
    
    
    def get_decomp(H, lowr):
    
        totp = H.shape[0]
        U, S, V = randomized_svd(H, lowr) 
        D = np.diag(H - U * S @ U.T)
        Hnew = np.diag(D) + U * S @ U.T
        diff = np.linalg.norm(Hnew - H, 2) / np.linalg.norm(H, 2)
    
        return D, U * (S**(1/2)), diff
    
    totp, num_cout = W.shape
    update_iter = int(update_iter)
    
    W_vec = np.copy(W).reshape(-1)
    W_sort = np.argsort(-np.abs(W_vec))
    W_vec[W_sort[k_spar:]] = 0
    
    W_sol = W_vec.reshape(totp, num_cout)
    
    XTX_inv = np.linalg.inv(XTX)
    D, L, _ = get_decomp(np.copy(XTX), int(XTX.shape[0] * low_ratio))
    lowr = L.shape[1]
    Dinv = 1 / D
    Cinv = np.linalg.inv(np.eye(lowr) + (L.T * Dinv) @ L)
    Linv = (L.T * Dinv).T
    
    Hess_diag = np.zeros((totp,))
    for i1 in range(totp):
        Hess_diag[i1] = Dinv[i1] - Linv[i1,:] @ Cinv @ Linv[i1,:].T 
    
    for i_c in range(num_cout):
        
        mask = np.where(W_sol[:,i_c] != 0)[0] 
        k_sp = len(mask)
        
        #w_prune = (XTY[:,i_c] * Dinv) - (Linv @ (Cinv @ (Linv.T @ XTY[:,i_c])))
        w_prune = XTX_inv @ XTY[:,i_c]
        
        k_init = np.count_nonzero(w_prune)
        mask_idx = np.where(w_prune != 0, 0, np.inf)
        
        Cinv_cur = np.copy(Cinv)
        Diag_cur = np.copy(Hess_diag)
        Linv_cur = np.copy(Linv)
        
        for i_t in range(k_init - k_sp):
             
            prune_idx = np.argmin( (w_prune)**2 / (Diag_cur + mask_idx) + mask_idx)
            
            mask_idx[prune_idx] = np.inf
            
            Hinv_band = - Linv_cur @ (Cinv_cur @ Linv_cur[prune_idx,:].T) 
            Hinv_band[prune_idx] += Dinv[prune_idx]     
            
            w_prune -= Hinv_band * (w_prune[prune_idx] / Diag_cur[prune_idx])
            w_prune[prune_idx] = 0
            
            CL = Cinv_cur @ Linv_cur[prune_idx,:].T
            Einv = 1 / (Dinv[prune_idx] - Linv_cur[prune_idx,:] @ CL)
            
            for i_n in range(totp):
                if mask_idx[i_n] == np.inf:
                    continue
                
                LCL = Linv_cur[i_n,:] @ CL
                Diag_cur[i_n] -= LCL * Einv * LCL  
            
            Cinv_cur = Cinv_cur + (Einv * CL)[:,np.newaxis] @ CL[np.newaxis,:]
            Linv_cur[prune_idx,:] = 0
            
            if (i_t+1)% update_iter == 0:
                
                for i_cd in range(totp): 
                    if mask_idx[i_cd] < 1:
                        w_prune[i_cd] = (XTY[i_cd,i_c]-XTX[i_cd,:]@w_prune + XTX[i_cd,i_cd] * w_prune[i_cd]) / XTX[i_cd,i_cd]
            
                        
        W_sol[:,i_c] = np.copy(w_prune)
            
    return weight_update_unstr(W_sol, XTX, XTY)


def backward_selection_all_unstr(W, XTX, XTY, k_spar):

    totp, num_cout = W.shape
    W_sol = np.zeros_like(W)
    
    XTX_inv = np.linalg.inv(XTX)
    XTX_invlist = np.zeros((XTX.shape[0], XTX.shape[1], num_cout))
    for i_c in range(num_cout):
        XTX_invlist[:,:,i_c] = np.copy(XTX_inv)
        
    W_sol = XTX_inv @ XTY
    mask_idx = np.where(W_sol != 0, 0, np.inf)
    prune_obj = np.zeros_like(W_sol)
    for i_c in range(num_cout):
        prune_obj[:,i_c] = (W_sol[:,i_c])**2 / (np.diag(XTX_invlist[:,:,i_c]) + mask_idx[:,i_c]) + mask_idx[:,i_c]
    
    k_init = np.count_nonzero(W_sol)
    for i_p  in range(k_init - k_spar):
        
        prune_idx = np.argmin(prune_obj)
        idx1 = prune_idx // num_cout
        idx2 = prune_idx % num_cout
        
        mask_idx[idx1, idx2] = np.inf
        W_sol[:,idx2] -= (W_sol[idx1,idx2] /  XTX_invlist[idx1,idx1,idx2]) * XTX_invlist[:,idx1,idx2] 
        W_sol[idx1,idx2] = 0
        
        XTX_invlist[:,:,idx2] -= XTX_invlist[:,idx1:idx1+1,idx2] @ (XTX_invlist[idx1:idx1+1,:,idx2] / XTX_invlist[idx1,idx1,idx2])
        
        XTX_invlist[:,idx1,idx2] = 0
        XTX_invlist[idx1,:,idx2] = 0
        
        prune_obj[:,idx2] = (W_sol[:,idx2])**2 / (np.diag(XTX_invlist[:,:,idx2]) + mask_idx[:,idx2]) + mask_idx[:,idx2]
        
    return W_sol, np.sum( -W_sol * XTY + (1/2) * W_sol * (XTX@W_sol) ) 

def greedy_CD_all_unstr(W, XTX, XTY, k_spar, update_iter = 1):
    
    if update_iter >= 1:
        i_repeat = 1
        update_iter = int(update_iter)
    else:
        i_repeat = int(1/update_iter)
        update_iter = 1
    
    totp, num_cout = W.shape
    
    update_list = np.zeros((num_cout,))
    XTX_inv = np.linalg.inv(XTX)
    W_sol = XTX_inv @ XTY
    
    mask_idx = np.where(W_sol != 0, 0, np.inf)
    obj_sum = np.zeros_like(W_sol)
    for i_c in range(num_cout):
        obj_sum[:,i_c] = W_sol[:,i_c] * (-XTX@W_sol[:,i_c] + XTY[:,i_c] + 0.5 * np.diag(XTX) * W_sol[:,i_c])
    
    k_init = np.count_nonzero(W_sol)
    for i_p  in range(k_init - k_spar):
        
        prune_idx = np.argmin(obj_sum + mask_idx)
        idx1 = prune_idx // num_cout
        idx2 = prune_idx % num_cout
        
        w_tmp = W_sol[idx1, idx2]
        mask_idx[idx1, idx2] = np.inf
        W_sol[idx1, idx2] = 0
        
        update_list[idx2] += 1
        

        if (update_list[idx2] + 1) % update_iter == 0:
                
            for _ in range(i_repeat):
                for i_cd in range(totp):   
                    if mask_idx[i_cd, idx2] < 1:
                        W_sol[i_cd, idx2] = (XTY[i_cd,idx2]- XTX[i_cd,:]@W_sol[:, idx2] + XTX[i_cd,i_cd] * W_sol[i_cd, idx2]) / XTX[i_cd,i_cd]
                
            obj_sum[:,idx2] = W_sol[:,idx2] * (-XTX@W_sol[:,idx2] + XTY[:,idx2] + 0.5 * np.diag(XTX) * W_sol[:,idx2])
            
        else:
            obj_sum[:,idx2] += w_tmp * (W_sol[:,idx2] * XTX[:,idx1])
            obj_sum[idx1,idx2] = 0
                

    return weight_update_unstr(W_sol, XTX, XTY)

def greedy_CD2_all_unstr(W, XTX, XTY, k_spar, update_iter = 1):
    
    if update_iter >= 1:
        i_repeat = 1
        update_iter = int(update_iter)
    else:
        i_repeat = int(1/update_iter)
        update_iter = 1
    
    totp, num_cout = W.shape
    
    update_list = np.zeros((num_cout,))
    XTX_inv = np.linalg.inv(XTX)
    W_sol = XTX_inv @ XTY
    
    mask_idx = np.where(W_sol != 0, 0, np.inf)
    Hdiag = np.zeros_like(W_sol) 
    for i_c in range(num_cout):
        Hdiag [:,i_c] = np.copy(np.diag(XTX_inv))
    
    k_init = np.count_nonzero(W_sol)
    for i_p  in range(k_init - k_spar):
        
        prune_idx = np.argmin(W_sol **2 / Hdiag + mask_idx)
        idx1 = prune_idx // num_cout
        idx2 = prune_idx % num_cout
        
        mask_idx[idx1, idx2] = np.inf
        W_sol[idx1, idx2] = 0
        
        Hdiag[:,idx2] = Hdiag[:,idx2] - XTX_inv[idx1,:] **2 / Hdiag[idx1,idx2]
        Hdiag[:,idx2] = np.maximum(Hdiag[:,idx2], 1/np.diag(XTX))
        
        update_list[idx2] += 1
        
        if (update_list[idx2] + 1) % update_iter == 0:
                
            for _ in range(i_repeat):
                for i_cd in range(totp):   
                    if mask_idx[i_cd, idx2] < 1:
                        W_sol[i_cd, idx2] = (XTY[i_cd,idx2]- XTX[i_cd,:]@W_sol[:, idx2] + XTX[i_cd,i_cd] * W_sol[i_cd, idx2]) / XTX[i_cd,i_cd]


    return weight_update_unstr(W_sol, XTX, XTY)

def backward_lowrank_all_unstr(W, XTX, XTY, k_spar, low_ratio, update_iter):
    
    
    def get_decomp(H, lowr):
    
        totp = H.shape[0]
        U, S, V = randomized_svd(H, lowr) 
        D = np.diag(H - U * S @ U.T)
        Hnew = np.diag(D) + U * S @ U.T
        diff = np.linalg.norm(Hnew - H, 2) / np.linalg.norm(H, 2)
    
        return D, U * (S**(1/2)), diff
    
    totp, num_cout = W.shape
    update_iter = int(update_iter)
    
    XTX_inv = np.linalg.inv(XTX)
    W_sol = XTX_inv @ XTY
    D, L, _ = get_decomp(np.copy(XTX), int(XTX.shape[0] * low_ratio))
    lowr = L.shape[1]
    Dinv = 1 / D
    Cinv = np.linalg.inv(np.eye(lowr) + (L.T * Dinv) @ L)
    Linv = (L.T * Dinv).T
    
    
    
    Hess_diag = np.zeros((totp,num_cout))
    for i1 in range(totp):
        Hess_diag[i1,:] = Dinv[i1] - Linv[i1,:] @ Cinv @ Linv[i1,:].T 
        
    Cinv_list = np.zeros((lowr,lowr,num_cout))
    Linv_list = np.zeros((totp,lowr,num_cout))
    for i_c in range(num_cout):
        Cinv_list[:,:,i_c] = np.copy(Cinv)
        Linv_list[:,:,i_c] = np.copy(Linv)

    update_list = np.zeros((num_cout,))
    
    mask_idx = np.where(W_sol != 0, 0, np.inf)
    k_init = np.count_nonzero(W_sol)
    
    
    prune_obj = np.zeros_like(W_sol)
    for i_c in range(num_cout):
        prune_obj[:,i_c] = (W_sol[:,i_c])**2 / (Hess_diag[:,i_c] + mask_idx[:,i_c]) + mask_idx[:,i_c]
    
    
    
    for i_p  in range(k_init - k_spar):
        
        prune_idx = np.argmin(prune_obj)
        idx1 = prune_idx // num_cout
        idx2 = prune_idx % num_cout
        
        mask_idx[idx1,idx2] = np.inf
            
        Hinv_band = - Linv_list[:,:,idx2] @ (Cinv_list[:,:,idx2] @ Linv_list[idx1,:,idx2].T) 
        Hinv_band[idx1] += Dinv[idx1]     
            
        W_sol[:,idx2] -= Hinv_band * (W_sol[idx1,idx2] / Hess_diag[idx1,idx2])
        W_sol[idx1,idx2] = 0
            
        CL = Cinv_list[:,:,idx2] @ Linv_list[idx1,:,idx2].T
        Einv = 1 / (Dinv[idx1] - Linv_list[idx1,:,idx2] @ CL)
            
        for i_n in range(totp):
            if mask_idx[i_n,idx2] == np.inf:
                continue
                
            LCL = Linv_list[i_n,:,idx2] @ CL
            Hess_diag[i_n,idx2] -= LCL * Einv * LCL  
            
        Cinv_list[:,:,idx2] += (Einv * CL)[:,np.newaxis] @ CL[np.newaxis,:]
        Linv_list[idx1,:,idx2] = 0
        
        
        update_list[idx2] += 1
        if (update_list[idx2] + 1) % update_iter == 0:
                
            for i_cd in range(totp):   
                if mask_idx[i_cd, idx2] < 1:
                    W_sol[i_cd, idx2] = (XTY[i_cd,idx2]- XTX[i_cd,:]@W_sol[:, idx2] + XTX[i_cd,i_cd] * W_sol[i_cd, idx2]) / XTX[i_cd,i_cd]
                    
        prune_obj[:,idx2] = (W_sol[:,idx2])**2 / (Hess_diag[:,idx2] + mask_idx[:,idx2]) + mask_idx[:,idx2]
                    
    return weight_update_unstr(W_sol, XTX, XTY)


def ADMM_pruner(W, XTX, XTY, k_spar, rho, max_iter = 1000, update_iter = 10, verbose = False):
    
    totp, num_cout = W.shape
    init_rho = False
    
    XTX_inv = np.linalg.inv(XTX + rho * np.eye(totp))
    
    V = np.zeros_like(W)
    
    D = np.copy(W).reshape(-1)
    loss_idx = np.argsort(-D**2)
    D[loss_idx[k_spar:]] = 0    
    D_suppp = (D == 0).astype('float')
    D = D.reshape(totp, num_cout)
    
    for i_admm in range(max_iter):
        
        
        W = XTX_inv @ (XTY-V+rho*D)
        
        D = ((V + rho * W) / rho).reshape(-1)
        loss_idx = np.argsort(-D**2)
        D[loss_idx[k_spar:]] = 0
        D = D.reshape(totp, num_cout)        
        V = V + rho * (W - D)

    
        
        if (i_admm+1) % update_iter == 0:
            
            D_supp = (D.reshape(-1) == 0).astype('float')
            supp_change = np.sum((D_supp-D_suppp)**2)
            
            if supp_change / k_spar > 0.5:
                init_rho = True
                rho *= 1.4
            elif supp_change / k_spar > 0.25:
                init_rho = True
                rho *= 1.2
            elif supp_change > 0.5:
                if init_rho:
                    rho *= 1.1
                else:
                    rho /= 5
            else:
                if init_rho:
                    break
                else:
                    rho /= 5
            
            D_suppp = np.copy(D_supp)
            XTX_inv = np.linalg.inv(XTX + rho * np.eye(totp))
            if verbose:
                print("loss is", np.sum( -W * XTY + (1/2) * W * (XTX@W))+ np.sum(V * (W-D)) + rho * np.sum((W-D)**2) ,
                      "change is ",supp_change, "new rho is",rho)
    W = W.reshape(-1)
    loss_idx = np.argsort(-W**2)
    W[loss_idx[k_spar:]] = 0
    W = W.reshape(totp, num_cout)
    
    return weight_update_unstr(W, XTX, XTY)



