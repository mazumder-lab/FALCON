from .utils import *
import time
from pyhessian.hessian import hessian #Needed for experiments with scaling the first order term
from utils.flops_utils import get_flops
from contextlib import nullcontext
import copy
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
IHTPATH = './Lagrangian-Heuristic'
sys.path.append(IHTPATH)
from group_prunealg import get_blocks, find_module, find_all_module, channel_selection, greedy_CD, greedy_CD_v2, greedy_CD_v3, greedy_CD_v4, greedy_CD_hf, forward_CD, backward_selection, backward_lowrank, backward_lowrank2, backward_lowrank3, weight_selection, greedy_CD_unstr, greedy_CD2_unstr, greedy_CD_all_unstr, greedy_CD2_all_unstr, backward_selection_unstr, backward_lowrank_unstr, backward_selection_all_unstr, backward_lowrank_all_unstr, ADMM_pruner, geomedian_selection, local_search, greedy_thi, group_LASSO, refine_LASSO,  local_search_v3

class LayerPruner:

    def __init__(self,model,params, train_dataset, train_dataloader, test_dataloader,
    nsamples,criterion,lambda_inv,gradseed,
    device,algo,lowr,update_iter, graditer, lr, L_parallel, N_sparsity):

        self.model = model
        self.params = params 
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.nsamples = nsamples
        self.lambda_inv = lambda_inv #compute inverse of H + lambda I
        self.device = device 
        self.algo = algo
        self.lowr = lowr
        self.update_iter = update_iter
        self.grads = None
        self.results = dict()
        self.gradseed = gradseed
        self.graditer = graditer
        self.lr = lr
        self.parallel = L_parallel
        self.nonuniform = N_sparsity

    def update_model(self,new_w):
        set_pvec(new_w, self.model,self.params,self.device)

    def compute_flops(self,input_res):
        self.model.eval()
        self.results['flops'] = get_flops(input_res,self.device,self.model)

    def reset_pruner(self):
        self.results = dict()
        self.grads=None
    
    def getinput_hook(self, module, input, output):
        self.input_buffer.append(input[0].detach())
        return

    def get_size(self, use_layer = True):
        size_list = []
        ignore_bias = True
        for name, param in self.model.named_parameters():
            layer_name,param_name = '.'.join(name.split('.')[:-1]),name.split('.')[-1]
            if ignore_bias == True and param_name == 'bias':
                continue
            if use_layer:
                if (not self.layerparams is None) and (not name in self.layerparams):
                    continue
                param_size = np.prod(param.shape)
                size_list.append(param.shape)
            else:
                if (not self.params is None) and (not name in self.params):
                    continue
                param_size = np.prod(param.shape)
                size_list.append(param.shape)
        
        self.size_list = size_list
    
    def get_params(self):
        if self.model.name == 'ResNetCifar':
            self.datasize = [3,32,32]
            self.layerparams = ['layer1.0.conv2.weight', 'layer1.1.conv2.weight', 'layer1.2.conv2.weight',
                     'layer2.0.conv2.weight', 'layer2.1.conv2.weight', 'layer2.2.conv2.weight',
                     'layer3.0.conv2.weight', 'layer3.1.conv2.weight', 'layer3.2.conv2.weight']
            
            self.sparsity_dict = {50:[0.62, 0.62, 0.62, 0.372, 0.558, 0.558, 0.372, 0.372, 0.372],
                                  40:[0.62, 0.775, 0.62, 0.202, 0.403, 0.403, 0.202, 0.151, 0.151],
                                  30:[0.373, 0.746, 0.746, 0.093, 0.046, 0.232, 0.14, 0.046, 0.186],
                                  20:[0.336, 0.168, 0.504, 0.168, 0.056, 0.28, 0.078, 0.078, 0.097],
                                  10:[0.196, 0.049, 0.244, 0.118, 0.147, 0.029, 0.012, 0.05, 0.037]}
            
            
        elif self.model.name == 'MobileNet':
            self.datasize = [3,224,224]
            self.layerparams = ['model.1.3.weight','model.2.3.weight','model.3.3.weight','model.4.3.weight','model.5.3.weight',
                     'model.6.3.weight','model.7.3.weight','model.8.3.weight','model.9.3.weight','model.10.3.weight',
                     'model.11.3.weight','model.12.3.weight','model.13.3.weight']
        elif self.model.name == 'ResNet':
            self.datasize = [3,224,224]
            self.layerparams = ['layer1.0.conv2.weight','layer1.0.conv3.weight','layer1.1.conv2.weight','layer1.1.conv3.weight',
                     'layer1.2.conv2.weight','layer1.2.conv3.weight','layer2.0.conv2.weight','layer2.0.conv3.weight',
                     'layer2.1.conv2.weight','layer2.1.conv3.weight','layer2.2.conv2.weight','layer2.2.conv3.weight',
                     'layer2.3.conv2.weight','layer2.3.conv3.weight','layer3.0.conv2.weight','layer3.0.conv3.weight',
                     'layer3.1.conv2.weight','layer3.1.conv3.weight','layer3.2.conv2.weight','layer3.2.conv3.weight',
                     'layer3.3.conv2.weight','layer3.3.conv3.weight','layer3.4.conv2.weight','layer3.4.conv3.weight',
                     'layer3.5.conv2.weight','layer3.5.conv3.weight','layer4.0.conv2.weight','layer4.0.conv3.weight',
                     'layer4.1.conv2.weight','layer4.1.conv3.weight','layer4.2.conv2.weight','layer4.2.conv3.weight']

    def prune_group(self,sparsity):

        zero_grads(self.model)
        self.model.eval()
        self.get_params()
        self.get_size()
        
        original_weight = get_pvec(self.model, self.layerparams)
        w_layer = original_weight.to('cpu').numpy()
        
        print('Starting Optimization')
        
        w_prune = np.copy(w_layer)
  
        tot_time, tot_time2, tot_time3 = 0, 0, 0
        
        torch.manual_seed(self.gradseed)
        torch.cuda.manual_seed(self.gradseed)
        torch.cuda.manual_seed_all(self.gradseed)
        np.random.seed(self.gradseed)
        train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=True,num_workers=10,pin_memory=True)
        
        start_data = time.time()
        xdata = torch.zeros([self.nsamples]+self.datasize).to("cuda")
        # collect data
        for i, batch in enumerate(train_dataloader):
    
            xdata_i, ydata_i = batch
            xdata_i = xdata_i.to("cuda")
            xdata[i,:,:,:] = xdata_i
            if (i + 1) % self.nsamples == 0:
                break
        
        xdata2 = copy.deepcopy(xdata)
        tot_time2 += time.time() - start_data
        
        
        i_w = 0
        i_layer = 0   
        block_list = get_blocks(copy.deepcopy(self.model))
        
        model_update = copy.deepcopy(self.model)
        update_dict = model_update.state_dict()
        
        if self.nonuniform:
            sparsity_list = self.sparsity_dict[int(100*sparsity)]
        else:
            sparsity_list = [sparsity for i in range(len(self.layerparams))]
        
        for name, block in block_list:
            
            with torch.no_grad():

                start_data = time.time()
                print("name is ",name)
                
                prune_list = find_all_module(block, self.layerparams, name, []) 
                if not prune_list:
                    xdata = block(xdata)
                    xdata2 = block(xdata2)
                    continue
            
            block_update = copy.deepcopy(block)    
            for cur_i in range(len(prune_list)):
                
                with torch.no_grad():
                    
                    # get sparsity
                    sparsity = sparsity_list[i_layer]
            
                    cur_module = prune_list[cur_i]
                    start_data = time.time()
                    prune_flag, prune_module = find_module(block_update, cur_module, name) 

                    hook_handle = prune_module.register_forward_hook(self.getinput_hook)
                    unfold = nn.Unfold(prune_module.kernel_size,dilation=prune_module.dilation,
                                          padding=prune_module.padding,stride=prune_module.stride)
        
                    self.input_buffer = []
                    block_update(xdata)
                    inp = np.vstack([unfold(inss).permute([1, 0, 2]).flatten(1).to("cpu").numpy() for inss in self.input_buffer])
                    hook_handle.remove()
                    prune_flag, prune_module = find_module(block, cur_module, name)
            
                    hook_handle = prune_module.register_forward_hook(self.getinput_hook)
                    self.input_buffer = []
                    block(xdata2)
                    inp2 = np.vstack([unfold(inss).permute([1, 0, 2]).flatten(1).to("cpu").numpy() for inss in self.input_buffer])
                    hook_handle.remove()
                    tot_time2 += time.time() - start_data
        
                    param_size = self.size_list[i_layer]
                    count = np.prod(param_size)
                    w_var = np.copy(w_layer[i_w:i_w+count]).reshape(param_size[0],-1).T
        
                    if self.parallel:
                        XTX = inp2 @ inp2.T / self.nsamples
                    else:
                        XTX = inp @ inp.T / self.nsamples
                    XTX += self.lambda_inv * np.eye(XTX.shape[0]) * np.mean(np.diag(XTX))
                    XTY = (inp @ inp2.T) @ w_var / self.nsamples
                    
                    start_algo = time.time()
                    if self.algo == "MP":
            
                        w_sol, w_obj = weight_selection(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)))
                
                    if self.algo == "GEO":
            
                        w_sol, w_obj = geomedian_selection(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)))
                
                    if self.algo == "IHT":
            
                        w_sol, w_obj = channel_selection(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)))
            
                    elif self.algo == "CDv1":
                
                        w_sol, w_obj = greedy_CD(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)), self.update_iter)
            
                    elif self.algo == "CDv2": 
                
                        w_sol, w_obj = greedy_CD_v2(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)), self.update_iter)
                    
                    elif self.algo == "CDv3": 
                
                        w_sol, w_obj = greedy_CD_hf(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)), self.update_iter)
                    
                    elif self.algo == "CDv4": 
                
                        w_sol, w_obj = forward_CD(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)), self.update_iter)
                    
                    elif self.algo == "CDv5": 
                        
                        w_sol, w_obj = greedy_CD_v3(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)), self.update_iter)
                       
                    elif self.algo == "CDv6": 
                        
                        w_sol, w_obj = greedy_CD_v4(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)), self.update_iter)
                    
                    elif self.algo == "Greedy": 
                
                        w_sol, w_obj = greedy_thi(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)))
                    
                    elif self.algo == "LASSO": 
                        
                        lam1 = 0.005 * np.sum(np.diag(XTX))
                        w_sol, w_obj = group_LASSO(w_var, XTX, XTY, lam1, 0, np.copy(w_var), param_size[1], 100)
                        w_sol, w_obj = refine_LASSO(w_sol, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)))
                    
                    elif self.algo == "Local": 
                
                        w_sol, w_obj = greedy_CD(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)), self.update_iter)
                        w_sol, w_obj = local_search(np.copy(w_sol), XTX, XTY, param_size[1], 20, 5)
                        
                    elif self.algo == "Local2": 
                
                        w_sol, w_obj = greedy_CD_v3(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)), self.update_iter)
                        w_sol, w_obj = local_search_v3(np.copy(w_sol), XTX, XTY, param_size[1], max_iter = 30, switch_ratio = 0.05, switch_lb = 0.001)
                        
                    elif self.algo == "Local3": 
                
                        w_sol, w_obj = greedy_CD_v4(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)), self.update_iter)
                        w_sol, w_obj = local_search_v3(np.copy(w_sol), XTX, XTY, param_size[1], max_iter = 30, switch_ratio = 0.05, switch_lb = 0.001)
                        
                    elif self.algo == "Local-IHT": 
                
                        w_sol, w_obj = channel_selection(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)))
                        w_sol, w_obj = local_search(np.copy(w_sol), XTX, XTY, param_size[1], 20, 5)
            
                    elif self.algo == "Back":

                        w_sol, w_obj = backward_selection(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)))
                
                    elif self.algo == "Backlow":
                
                        w_sol, w_obj = backward_lowrank(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)), self.lowr)
                    
                    elif self.algo == "Backlow2":
                
                        w_sol, w_obj = backward_lowrank2(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)), self.lowr)
                    
                    elif self.algo == "Backlow3":
                
                        w_sol, w_obj = backward_lowrank3(w_var, XTX, XTY, param_size[1], int(param_size[1] * (1-sparsity)), self.lowr)

                    tot_time += time.time() - start_algo
            
               
                    w_output = np.copy((w_sol.T).reshape(-1))
                    w_prune[i_w:i_w+count] = np.copy((w_sol.T).reshape(-1))
                    i_w += count
                    i_layer += 1
                    start_data = time.time()
                    state_dict = block_update.state_dict()
                    if cur_module.startswith(name+"."):
                        param_local = copy.deepcopy(cur_module[len(name+"."):])
                    else:
                        param_local = copy.deepcopy(cur_module)
                    state_dict[param_local] = torch.Tensor(w_output).to("cuda").reshape(state_dict[param_local].shape)
                
                    block_update.load_state_dict(state_dict)
                    
                    tot_time2 += time.time() - start_data
            
            ydata2 = block(xdata2).detach()
            ysum = torch.sum(ydata2**2)
            
            optimizer = torch.optim.SGD(block_update.parameters(), lr=self.lr)

            start_grad = time.time()
            for i_grad in range(self.graditer):

                optimizer.zero_grad()
                     
                ydata = block_update(xdata)
                loss = torch.sum((ydata-ydata2)**2)  / ysum
                loss.backward()
                for name2, param2 in block_update.named_parameters():
                    if not 'conv' in name2:
                        param2.grad = None
                    else:
                        param2.grad *= (param2 != 0)
       
                optimizer.step()
    
                if (i_grad+1) % 200 == 0:
                    print("layer",i_layer,"loss",(torch.sum((ydata-ydata2)**2)  / ysum).detach().to("cpu").numpy())

            print("cost",time.time()-start_grad)
            tot_time3 += time.time()-start_grad
            
            start_data = time.time()  
            with torch.no_grad():            
                xdata = block_update(xdata)
                xdata2 = block(xdata2)
            
                state_dict = block_update.state_dict()
                update_list = find_all_module(block_update, self.params, name, []) 
                for upd_i in range(len(update_list)):

                    upd_module = update_list[upd_i]
                
                    if upd_module.startswith(name+"."):
                        param_local = copy.deepcopy(upd_module[len(name+"."):])
                    else:
                        param_local = copy.deepcopy(upd_module)
                
                    update_dict[upd_module] = copy.deepcopy(state_dict[param_local])
            tot_time2 += time.time() - start_data
 
        model_update.load_state_dict(update_dict)
    
        #model_new3 = copy.deepcopy(self.model)
        #set_pvec(w_prune, model_new3, self.layerparams,"cuda")

        propagate_sparsity(model_update,DataLoader(self.train_dataset, batch_size=1, shuffle=True,num_workers=40,pin_memory=True), self.criterion,samples=100)
        w_prunedL2 = get_pvec(model_update, self.params).to('cpu').numpy().astype(np.float64)
        
        final_acc = compute_acc(model_update,self.test_dataloader,self.device)
        print(final_acc)
        print(tot_time,tot_time2,tot_time3)
        print(1 - np.count_nonzero(w_prunedL2) / len(w_prunedL2))
        self.results['sparsity']=(sparsity)
        self.results['sparsity_true']= 1 - np.count_nonzero(w_prunedL2) / len(w_prunedL2)
        self.results['lowr']= self.lowr
        self.results['update'] = self.update_iter
        self.results['prun_runtime']=tot_time
        self.results['data_runtime']=tot_time2
        self.results['grad_runtime']=tot_time3
        self.results['norm_w']=(np.linalg.norm(w_prunedL2,ord=2))
        self.results['flop'] = np.sum((w_prunedL2 !=0) * generate_weight(self.model.name))
        self.results['test_acc'] = final_acc
        self.results['graditer'] = self.graditer
        self.results['lr'] = self.lr
        
    
    def prune_unstr(self,sparsity):

        zero_grads(self.model)
        self.model.eval()
        self.get_params()
        self.get_size(False)
        
        original_weight = get_pvec(self.model, self.params)
        w_layer = original_weight.to('cpu').numpy()
        
        print('Starting Optimization')
        
        w_prune = np.copy(w_layer)
  
        tot_time, tot_time2 = 0, 0
        
        torch.manual_seed(self.gradseed)
        torch.cuda.manual_seed(self.gradseed)
        torch.cuda.manual_seed_all(self.gradseed)
        np.random.seed(self.gradseed)
        train_dataloader = DataLoader(self.train_dataset, batch_size=1, shuffle=True,num_workers=10,pin_memory=True)
        
        start_data = time.time()
        xdata = torch.zeros([self.nsamples]+self.datasize).to("cuda")
        # collect data
        for i, batch in enumerate(train_dataloader):
    
            xdata_i, ydata_i = batch
            xdata_i = xdata_i.to("cuda")
            xdata[i,:,:,:] = xdata_i
            if (i + 1) % self.nsamples == 0:
                break
        
        xdata2 = copy.deepcopy(xdata)
        tot_time2 += time.time() - start_data
        
        
        i_w = 0
        i_layer = 0   
        block_list = get_blocks(copy.deepcopy(self.model))
            
        for name, block in block_list:
            
            with torch.no_grad():

                start_data = time.time()
                print("name is ",name)
                
                prune_list = find_all_module(block, self.params, name, []) 
                if not prune_list:
                    xdata = block(xdata)
                    xdata2 = block(xdata2)
                    continue
            
                block_update = copy.deepcopy(block)    
                for cur_i in range(len(prune_list)):
            
                    cur_module = prune_list[cur_i]
                    start_data = time.time()
                    prune_flag, prune_module = find_module(block_update, cur_module, name) 

                    hook_handle = prune_module.register_forward_hook(self.getinput_hook)
                    
                    self.input_buffer = []
                    block_update(xdata)
                    
                    if type(prune_module)  == nn.Conv2d:
                        unfold = nn.Unfold(prune_module.kernel_size,dilation=prune_module.dilation,
                                           padding=prune_module.padding,stride=prune_module.stride)
                        inp = np.vstack([unfold(inss).permute([1, 0, 2]).flatten(1).to("cpu").numpy() for inss in self.input_buffer])
                    else:
                        inp = np.vstack([inss.permute([1, 0]).to("cpu").numpy() for inss in self.input_buffer])

                    hook_handle.remove()
                    prune_flag, prune_module = find_module(block, cur_module, name)
            
                    hook_handle = prune_module.register_forward_hook(self.getinput_hook)
                    self.input_buffer = []
                    if cur_i == len(prune_list) - 1:
                        xdata2 = block(xdata2)
                    else:
                        block(xdata2)
                        
                    if type(prune_module)  == nn.Conv2d:
                        unfold = nn.Unfold(prune_module.kernel_size,dilation=prune_module.dilation,
                                           padding=prune_module.padding,stride=prune_module.stride)
                        inp2 = np.vstack([unfold(inss).permute([1, 0, 2]).flatten(1).to("cpu").numpy() for inss in self.input_buffer])
                    else:
                        inp2 = np.vstack([inss.permute([1, 0]).to("cpu").numpy() for inss in self.input_buffer])

                    hook_handle.remove()
                    tot_time2 += time.time() - start_data
        
                    param_size = self.size_list[i_layer]
                    count = np.prod(param_size)
                    w_var = np.copy(w_layer[i_w:i_w+count]).reshape(param_size[0],-1).T
                    
                      
                    XTX = inp @ inp.T / self.nsamples
                    XTX += self.lambda_inv * np.eye(XTX.shape[0]) * np.mean(np.diag(XTX))
                    XTY = (inp @ inp2.T) @ w_var / self.nsamples
                    
                    k_spar = int(w_var.size * (1-sparsity)) 
                    start_algo = time.time()
                    if self.algo == "MP":
            
                        w_sol, w_obj = weight_selection_unstr(w_var, XTX, XTY, k_spar)
            
                    elif self.algo == "CDv1":
                
                        w_sol, w_obj = greedy_CD_unstr(w_var, XTX, XTY, k_spar, self.update_iter)
            
                    elif self.algo == "CDv2": 
                
                        w_sol, w_obj = greedy_CD_all_unstr(w_var, XTX, XTY, k_spar, self.update_iter)
                    
                    elif self.algo == "CDv3": 
                
                        w_sol, w_obj = greedy_CD2_unstr(w_var, XTX, XTY, k_spar, self.update_iter)
                    
                    elif self.algo == "CDv4": 
                
                        w_sol, w_obj = greedy_CD2_all_unstr(w_var, XTX, XTY, k_spar, self.update_iter)
            
                    elif self.algo == "Back":
                
                        w_sol, w_obj = backward_selection_unstr(w_var, XTX, XTY, k_spar)
                
                    elif self.algo == "Back2":
                
                        w_sol, w_obj = backward_selection_all_unstr(w_var, XTX, XTY, k_spar)
                    
                    elif self.algo == "Backlow":
                
                        w_sol, w_obj = backward_lowrank_unstr(w_var, XTX, XTY, k_spar, self.lowr, self.update_iter)
                    
                    elif self.algo == "Backlow2":
                
                        w_sol, w_obj = backward_lowrank_all_unstr(w_var, XTX, XTY, k_spar, self.lowr, self.update_iter)
                    
                    elif self.algo == "ADMM":
                        #if type(prune_module)  == nn.Conv2d:
                        w_sol, w_obj = ADMM_pruner(w_var, XTX, XTY, k_spar, rho = 100 / self.nsamples)
                        #else:
                        #    w_sol, w_obj = backward_selection_all_unstr(w_var, XTX, XTY, k_spar)

                    tot_time += time.time() - start_algo
                    print("finish", "time is", time.time() - start_algo, "shape is",XTY.shape)
               
                    w_output = np.copy((w_sol.T).reshape(-1))
                    w_prune[i_w:i_w+count] = np.copy((w_sol.T).reshape(-1))
                    i_w += count
                    i_layer += 1
                    start_data = time.time()
                    state_dict = block_update.state_dict()
                    if cur_module.startswith(name+"."):
                        param_local = copy.deepcopy(cur_module[len(name+"."):])
                    else:
                        param_local = copy.deepcopy(cur_module)
                    state_dict[param_local] = torch.Tensor(w_output).to("cuda").reshape(state_dict[param_local].shape)
                
                    block_update.load_state_dict(state_dict)
                    if cur_i == len(prune_list) - 1:
                        xdata = block_update(xdata)
                    
                    tot_time2 += time.time() - start_data
 
        model_new3 = copy.deepcopy(self.model)
        set_pvec(w_prune, model_new3, self.params,"cuda")

        propagate_sparsity(model_new3,DataLoader(self.train_dataset, batch_size=1, shuffle=True,num_workers=40,pin_memory=True), self.criterion,samples=100)
        w_prunedL2 = get_pvec(model_new3, self.params).to('cpu').numpy().astype(np.float64)
        
        print(compute_acc(model_new3,self.test_dataloader,self.device))
        print(tot_time,tot_time2)
        print(1 - np.count_nonzero(w_prunedL2) / len(w_prunedL2))
        self.results['sparsity']=(sparsity)
        self.results['sparsity_true']= 1 - np.count_nonzero(w_prunedL2) / len(w_prunedL2)
        self.results['lowr']= self.lowr
        self.results['update'] = self.update_iter
        self.results['prun_runtime']=tot_time
        self.results['grad_runtime']=tot_time2
        self.results['norm_w']=(np.linalg.norm(w_prunedL2,ord=2))
        self.results['flop'] = np.sum((w_prunedL2 !=0) * generate_weight(self.model.name))
        self.results['test_acc'] = compute_acc(model_new3,self.test_dataloader,self.device) 
        self.results['graditer'] = self.graditer
        self.results['lr'] = self.lr
        
        
        
        
