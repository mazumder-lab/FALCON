from .utils import *
import time
from pyhessian.hessian import hessian #Needed for experiments with scaling the first order term
from utils.flops_utils import get_flops
from contextlib import nullcontext
import copy
from torch.utils.data import DataLoader
import sys
IHTPATH = './Lagrangian-Heuristic'
sys.path.append(IHTPATH)
from group_prunealg import get_layerobj, channel_selection, greedy_CD, greedy_CD_v2, backward_selection, backward_lowrank

class LayerPruner:

    def __init__(self,model,params, train_dataset, train_dataloader, test_dataloader,
    ngrads,criterion,lambda_inv,gradseed,
    device,algo,lowr):

        self.model = model
        self.params = params 
        self.prun_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.ngrads = ngrads
        self.lambda_inv = lambda_inv #compute inverse of H + lambda I
        self.device = device 
        self.algo = algo
        self.lowr = lowr
        self.grads = None
        self.results = dict()
        self.gradseed = gradseed

    def update_model(self,new_w):
        set_pvec(new_w, self.model,self.params,self.device)

    def compute_flops(self,input_res):
        self.model.eval()
        self.results['flops'] = get_flops(input_res,self.device,self.model)

    def reset_pruner(self):
        self.results = dict()
        self.grads=None
      
    def get_size(self):
        size_list = []
        ignore_bias = True
        for name, param in self.model.named_parameters():
            layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
            if ignore_bias == True and param_name == 'bias':
                continue
            if (not self.params is None) and (not name in self.params):
                continue
        
            param_size = np.prod(param.shape)
            size_list.append(param.shape)
        
        self.size_list = size_list
    
    def get_usesize(self):
        if self.model.name == 'ResNetCifar':
            self.use_size = [2,4,6,8,10,12,14,16,18]
            self.drop_down = True
        elif self.model.name == 'MobileNet':
            self.use_size = [2,4,6,8,10,12,14,16,18,20,22,24,26]
            self.drop_down = True
        elif self.model.name == 'ResNet':
            self.use_size = [2,3,5,6,8,9,11,12,14,15,17,18,20,21,23,24,26,27,29,30,32,33,35,36,38,39,41,42,44,45,47,48]
            self.drop_down = True
        
    def prune_group(self,sparsity):
        
        original_weight = get_pvec(self.model, self.params)
        w1 = original_weight.to('cpu').numpy()
        
        zero_grads(self.model)
        self.model.eval()
        self.get_size()
        self.get_usesize()
        
        print('Starting Optimization')
        
        w_prunedL = np.copy(w1)
        
        i_w = 0
        tot_obj, tot_time, tot_time2 = 0, 0, 0
        
        for si in range(len(self.size_list)):
    
            count = np.prod(self.size_list[si])
            if si not in self.use_size:
                i_w += count
                continue
            
            start_grad = time.time()
            hess_list, xty_list = get_layerobj(self.model, self.params, self.gradseed, self.ngrads, self.prun_dataset, w_prunedL, w1, 
                                           [si], drop_down = self.drop_down, batch_size = 8)
            tot_time2 += time.time() - start_grad
            
            param_size = self.size_list[si]
            
            w_cur = np.copy(w1[i_w:i_w+count])
            w_input = (w_cur.reshape(-1).reshape(param_size[0],-1)).T
            hess_list[0] += self.lambda_inv * np.eye(hess_list[0].shape[0])
            
            start_algo = time.time()
            if self.algo == "MP":
            
                w_sol, w_obj = channel_selection(w_input, hess_list[0], xty_list[0], param_size[1], int(param_size[1] * (1-sparsity)))
            
            elif self.algo == "CDv1":
                
                w_sol, w_obj = greedy_CD(w_input, hess_list[0], xty_list[0], param_size[1], int(param_size[1] * (1-sparsity)), 1)
            
            elif self.algo == "CDv2":
                
                w_sol, w_obj = greedy_CD_v2(w_input, hess_list[0], xty_list[0], param_size[1], int(param_size[1] * (1-sparsity)), 1)
            
            elif self.algo == "Back":
                
                w_sol, w_obj = backward_selection(w_input, hess_list[0], xty_list[0], param_size[1], int(param_size[1] * (1-sparsity)))
                
            elif self.algo == "Backlow":
                
                w_sol, w_obj = backward_lowrank(w_input, hess_list[0], xty_list[0], param_size[1], int(param_size[1] * (1-sparsity)), self.lowr)

            tot_obj += w_obj
            tot_time += time.time() - start_algo
                
            w_prunedL[i_w:i_w+count] =  (w_sol.T).reshape(-1)   
            i_w += count
            
            print(si,end=" ")
            
    
        w_prunedL[i_w:] = np.copy(w1[i_w:])
        

        model_new3 = copy.deepcopy(self.model)
        set_pvec(w_prunedL, model_new3, self.params,"cuda")
        propagate_sparsity(model_new3,DataLoader(self.prun_dataset, batch_size=1, shuffle=True,num_workers=40,pin_memory=True), self.criterion, samples=100)
        w_prunedL2 = get_pvec(model_new3, self.params).to('cpu').numpy().astype(np.float64)
        
        
        self.results['norm_w_wbar']=(np.linalg.norm(w_prunedL2-w1,ord=2))
        self.results['sparsity']=(sparsity)
        self.results['sparsity_true']= 1 - np.count_nonzero(w_prunedL2) / len(w1)
        self.results['obj']= tot_obj
        self.results['lowr']= self.lowr
        self.results['prun_runtime']=tot_time
        self.results['grad_runtime']=tot_time2
        self.results['norm_w']=(np.linalg.norm(w_prunedL2,ord=2))
        self.results['flop'] = np.sum((w_prunedL2 !=0) * generate_weight(self.model.name))
        self.results['test_acc'] = compute_acc(model_new3,self.test_dataloader,self.device)
        
    
        
        
        
        
        
