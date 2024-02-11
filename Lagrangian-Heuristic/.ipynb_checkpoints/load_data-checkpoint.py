import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import time
import logging
import math
import sys
import numpy as np
import copy
import os
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import random
import GPUtil

from torch.utils.data import DataLoader
from torchvision.models import resnet50 as torch_resnet50
from collections import OrderedDict

## The woodfisher repo folder
WOODFISHERPATH = '../WoodFisher/'
sys.path.append(WOODFISHERPATH)

from options import get_parser
from policies import Manager
from models.mlpnet import MlpNet
from utils.datasets import mnist_get_datasets
from utils_new import *

def load_Xw(model_name, subsample_size, batch_size, seed = 0):
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if model_name == "mnist":
        train_dataset,test_dataset = mnist_get_datasets('./dataset/')
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,num_workers=2)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=2)
        manager = return_model(0.9,modules_to_prune=None,test_dataset=test_dataloader,fisher_subsample_size=100,wf_pruner='woodfisherblock',arch='mlpnet')
        model =copy.deepcopy(manager.model)
        modules_to_prune = ['fc1._layer','fc2._layer','fc3._layer']
        criterion = torch.nn.functional.nll_loss
        X, w_bar = compute_X(model,criterion,train_dataloader,subsample_size,modules_to_prune = modules_to_prune,device="cpu")
        
    
    return X, w_bar, model, train_dataloader, test_dataloader, modules_to_prune

def check_imp(model, w_pruned, test_dataloader, modules_to_prune=None, ignore_bias=True):
    
    original_sparsity = evaluate_sparsity(model,modules_to_prune=modules_to_prune)
    original_acc = compute_acc(model,test_dataloader,device="cpu") #sets up bn stats
    
    model_new = copy.deepcopy(model)
    
    i_w = 0
    with torch.no_grad():
        for name, param in model_new.named_parameters():
            #print("name is {} and shape of param is {} \n".format(name, param.shape))
            layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
            if ignore_bias and param_name == 'bias':
                continue
            if (not modules_to_prune is None) and (not layer_name in modules_to_prune):
                continue
            param_size = np.prod(param.shape)
            param.copy_(torch.from_numpy(w_pruned[i_w:i_w+param_size]).reshape_as(param))
            i_w += param_size
    
    model_new.zero_grad()
    model_new.eval()
    new_sparsity = evaluate_sparsity(model_new,modules_to_prune=modules_to_prune)
    new_acc = compute_acc(model_new,test_dataloader,device="cpu")
    
    return original_sparsity, original_acc, new_sparsity, new_acc
    
    

def compute_X(model,criterion,train_dataloader,fisher_subsample_size,modules_to_prune=None,ignore_bias=True,device='cpu'):
    #print('GPU Util begin')
    #GPUtil.showUtilization()
    i = 0
    gradients = []
    params = []
    for name, param in model.named_parameters():
        #print("name is {} and shape of param is {} \n".format(name, param.shape))
        layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
        if ignore_bias and param_name == 'bias':
            continue
        if (not modules_to_prune is None) and (not layer_name in modules_to_prune):
            continue
        params.append(param)
    w_bar = flatten_tensor_list(params)
    w_bar = w_bar.cpu()
    #print('GPU Util params')
    #GPUtil.showUtilization()
    for in_tensor, target in train_dataloader:
        in_tensor,target =in_tensor.to(device), target.to(device)
        output = model(in_tensor)
        loss = criterion(output, target)
        ys = loss
        grads = torch.autograd.grad(ys, params)

        del ys,loss,output

        grads = flatten_tensor_list(grads)
        grads=grads.cpu()
        model.zero_grad()
        gradients.append(grads)
        if i%100 == 0:
            print('---computing gradients-- ',i)
            #print('GPU Util during')
            #GPUtil.showUtilization()
        i+=1
        if i >= fisher_subsample_size:
            break
    del params
    X = torch.vstack(gradients)
    X = np.array(X.detach().cpu().numpy(),dtype=np.float64)
    w_bar = np.array(w_bar.detach().cpu().numpy(),dtype=np.float64)
    return X,w_bar

