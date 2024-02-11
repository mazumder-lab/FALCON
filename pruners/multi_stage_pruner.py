import re
import time
from .utils import *


class MultiStagePruner:
    def __init__(self,pruner,test_dataloader,schedule,num_stages,repeat=1):
        self.pruner = pruner
        self.test_dataloader = test_dataloader
        self.results = []
        self.schedule=schedule
        self.num_stages=num_stages
        self.repeat=repeat

    def reset_pruner(self):
        self.results=[]

    def get_input_dim(self):
        for i,o in self.test_dataloader:
            break
        return list(i.shape[1:])
    
    def prune(self,mask0,sparsity,base_level,grads=None):
        sparsities=generate_schedule(self.num_stages, base_level,sparsity,self.schedule,self.repeat)
        mask = torch.clone(mask0)
        input_dim = self.get_input_dim()
        if self.num_stages > 1:
            grads=None
        for i,sparsity_stg in enumerate(sparsities):
            self.pruner.reset_pruner()
            start = time.time()
            w_pruned,mask = self.pruner.prune(mask,sparsity_stg,grads=grads)
            end = time.time()
            print('Stage took', end-start)
            self.pruner.update_model(w_pruned)
            self.pruner.compute_flops(input_dim)
            self.results.append(self.pruner.results)
            if  not self.test_dataloader is None:
                self.results[-1]['test_acc'] = compute_acc(self.pruner.model,self.test_dataloader,self.pruner.device)
                print('Stage sp',sparsity_stg,self.results[-1]['test_acc'])
        return w_pruned,mask
    
    def prune_flop(self,mask0,sparsity,base_level,flop_ratio,flop_base, grads=None):
        
        sparsities=generate_schedule(self.num_stages, base_level,sparsity,self.schedule,self.repeat)
        flop_ratios=generate_schedule(self.num_stages, flop_base,flop_ratio,self.schedule,self.repeat)
        mask = torch.clone(mask0)
        input_dim = self.get_input_dim()
        if self.num_stages > 1:
            grads=None
        for i,sparsity_stg in enumerate(sparsities):
            self.pruner.reset_pruner()
            start = time.time()
            w_pruned,mask = self.pruner.prune_flop(mask,sparsity_stg,flop_ratios[i],grads=grads)
            end = time.time()
            print('Stage took', end-start)
            self.pruner.update_model(w_pruned)
            self.pruner.compute_flops(input_dim)
            self.results.append(self.pruner.results)
            
            if  not self.test_dataloader is None:
                self.results[-1]['flop_prune'] = np.sum((w_pruned !=0) * generate_weight(self.pruner.model.name))
                self.results[-1]['test_acc'] = compute_acc(self.pruner.model,self.test_dataloader,self.pruner.device)
                
                print('Stage sp',sparsity_stg,flop_ratios[i],self.results[-1]['test_acc'])
        return w_pruned,mask
    
    
    def prune_group(self,mask0,sparsity,base_level, grads=None):
        
        sparsities=generate_schedule(self.num_stages, base_level,sparsity,self.schedule,self.repeat)
        mask = torch.clone(mask0)
        input_dim = self.get_input_dim()
        if self.num_stages > 1:
            grads=None
        for i,sparsity_stg in enumerate(sparsities):
            self.pruner.reset_pruner()
            start = time.time()
            w_pruned,mask = self.pruner.prune_group(mask,sparsity_stg,grads=grads)
            end = time.time()
            print('Stage took', end-start)
            self.pruner.update_model(w_pruned)
            self.pruner.compute_flops(input_dim)
            self.results.append(self.pruner.results)
            
            if  not self.test_dataloader is None:
                self.results[-1]['flop_prune'] = np.sum((w_pruned !=0) * generate_weight(self.pruner.model.name))
                self.results[-1]['test_acc'] = compute_acc(self.pruner.model,self.test_dataloader,self.pruner.device)
                pruned,total=num_filters(self.pruner.model)
                self.results[-1]['pruned_filters'] = pruned
                self.results[-1]['total_filters'] = total
                pruned,total=num_channels(self.pruner.model)
                self.results[-1]['pruned_channels'] = pruned
                self.results[-1]['total_channels'] = total
                
                print('Stage sp',sparsity_stg,self.results[-1]['test_acc'])
        return w_pruned,mask