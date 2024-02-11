from .utils import *
import time
from pyhessian.hessian import hessian #Needed for experiments with scaling the first order term
from utils.flops_utils import get_flops
import autograd_hacks.autograd_hacks as autograd_hacks

class IHTCDLSPruner_torch:

    def __init__(self,model,params,prun_dataloader,
    ngrads,nsteps,fisher_mini_bsz,criterion,blocksize,lambda2,num_iterations,
    first_order_term,compute_trace_H,alpha_one,
    device,algo='Active_IHTCDLS',L=None):
        '''
         This object changes the model. 
        After prune is called, the attribute results is filled with the following keys :
        'norm_w_wbar','sparsity','new_non_zeros','trace_C','trace_H',
        'gradient_norm','obj','prun_runtime','norm_w'
        '''
        self.model = model
        self.params = params 
        self.prun_dataloader = prun_dataloader
        self.criterion = criterion
        self.ngrads = ngrads
        self.blocksize = blocksize
        self.l2 = lambda2 ##Un-normalized lambda2
        self.lambda2 = self.l2*ngrads/2 #self.lambda2 is the lambda in the regression formulation
        self.num_iterations = num_iterations 
        self.device = device 
        self.first_order_term =first_order_term
        self.compute_trace_H  = compute_trace_H
        self.alpha_one = alpha_one
        self.fisher_mini_bsz = fisher_mini_bsz
        self.algo = algo
        self.grads = None
        self.nsteps = nsteps
        self.results = dict()
        self.L=L
        self.mini_bsz = 0

        if self.blocksize > 0:
            #self.algo = 'Heuristic_LSBlock' #This is the only block IHT algorithm
            self.block_list = get_blocklist(self.model,self.params,self.blocksize)

    def update_model(self,new_w):
        set_pvec(new_w, self.model,self.params,self.device)

    def compute_flops(self,input_res):
        self.model.eval()
        self.results['flops'] = get_flops(input_res,self.device,self.model)

    def reset_pruner(self):
        self.results = dict()
        self.grads=None
        
    def prune(self,mask,sparsity,grads=None):
        for i, batch in enumerate(self.prun_dataloader):

            if (i) == self.nsteps*self.fisher_mini_bsz:
                break
            if i%10 == 0:
                print('-------> step',i)

            if (i) %self.fisher_mini_bsz == 0:
                self.grads = None

            original_weight = get_pvec(self.model, self.params)
            if mask is None:
                mask = torch.ones_like(original_weight).cpu() != 0
            w1 = original_weight
            d = len(w1)
            k = int((1-sparsity)*original_weight.numel())
            zero_grads(self.model)
            self.model.eval()
            #grads = torch.zeros((self.prun_dataloader.batch_size, d), device=self.device)
            start_grad_comp = time.time() 
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            loss = self.criterion(self.model(x), y)
            loss.backward(retain_graph=True)
            autograd_hacks.compute_grad1(self.model)
            grads = get_gvec1(self.model, self.params).to(self.device)
            zero_grads(self.model)
            autograd_hacks.clear_backprops(self.model)
            autograd_hacks.clear_activations(self.model)
            autograd_hacks.clear_grads(self.model)
            if grads.shape[0] != self.ngrads :
                print('Not enough samples')
                break
            end_grad_comp = time.time()
            #print('Grad computation  took ',end_grad_comp- start_grad_comp)
            self.model.eval()

            if self.grads is None:
                self.grads = grads
            else:
                self.grads += grads
            if (i+1) %self.fisher_mini_bsz == 0:
                self.grads /= self.fisher_mini_bsz
            else:
                continue
        
            y=grads@w1
            beta_tilde2=torch.clone(w1)
            trace_C = torch.linalg.norm(grads,ord='fro')**2/self.ngrads
            beta_tilde1 = torch.zeros_like(w1)


            ###########################
            if self.first_order_term:
                if self.alpha_one and not self.compute_trace_H:
                    trace_H = None
                    alpha = 1
                elif not self.compute_trace_H: ##Don't compute trace(H)
                    trace_H = None
                    alpha = 1/self.fisher_mini_bsz
                else:
                    start_hessian = time.time()
                    for inputs_trace,outputs_trace in self.prun_dataloader:
                        break
                    hessian_comp = hessian.hessian(self.model, self.criterion, data=(inputs_trace,outputs_trace),max_size=self.fisher_subsample_size, device=self.device)
                    trace = hessian_comp.trace()
                    trace_H = np.mean(trace)
                    print('Hessian computation took', time.time() - start_hessian)
                    alpha = (trace_C.cpu().item()/trace_H)
                    print('alpha =',alpha)

            else:
                trace_H = None 
                alpha = 0
            ###########################
            
            if alpha != 0:
                alpha_vec = alpha*grads.sum(axis=0) 
            else:
                alpha_vec = torch.zeros_like(w1,device=self.device)

            gradient_norm = torch.linalg.norm(grads.T@torch.ones(self.ngrads,device=self.device), ord=2) / self.ngrads

            
            start_algo = time.time()


            if self.algo == 'Active_IHTCDLS':
                w_pruned, obj, sols, _, _, sol_time = torch_algo.Active_IHTCDLS_PP(y,grads,w1,k,alpha=alpha_vec,lambda1=0,lambda2=self.lambda2,M=torch.inf, beta_tilde1=beta_tilde1,
                            beta_tilde2=beta_tilde2, L=None, iht_max_itr=self.num_iterations, ftol = 1e-7, act_max_itr=self.num_iterations, buget=None, kimp=1.5, act_itr=1,
                            cd_itr = 0, ctol = 1e-4, sea1_max_itr=5, sea2_max_itr=10)
                if 'Ls' in self.results:
                    self.results['Ls']+= [x.Ls for x in sols]
                else:
                    self.results['Ls']= [x.Ls for x in sols]
            elif self.algo == 'Active_IHTCDLS1':
                w_pruned, obj, _, _, _, sol_time = torch_algo.Active_IHTCDLS_PP(y,grads,w1,k,alpha=alpha_vec,lambda1=0,lambda2=self.lambda2,M=torch.inf, beta_tilde1=beta_tilde1,
                            beta_tilde2=beta_tilde2, L=None, iht_max_itr=0, ftol = 1e-7, act_max_itr=self.num_iterations, buget=None, kimp=1.5, act_itr=1,
                            cd_itr = 0, ctol = 1e-4, sea1_max_itr=5, sea2_max_itr=10)
            elif self.algo == 'CD':
                r = y - grads@w1
                S_diag = torch.linalg.norm(grads, axis=0)**2
                w_pruned, _ = torch_algo.CD_loop(y,grads,w1,r,S_diag,alpha_vec,0,self.lambda2,beta_tilde1,beta_tilde2,cd_itr=self.num_iterations,cd_tol = 1e-4)
            elif self.algo == 'Vanilla_IHTCDLS':
                w_pruned, obj, _, _, _,sol_time = torch_algo.Vanilla_IHTCDLS_PP(y,grads,w1,k,alpha=alpha_vec,lambda1=0,lambda2=self.lambda2,M=torch.inf, beta_tilde1=beta_tilde1,
                            beta_tilde2=beta_tilde2, L=None, iht_max_itr=self.num_iterations, ftol = 1e-7,
                            cd_itr = 0, ctol = 1e-4, sea_max_itr=5)
            elif self.algo == 'Vanilla_IHT':
                w_pruned, obj, _, _, _, sol_time = torch_algo.Vanilla_IHT(y,grads,w1,k,alpha=alpha_vec,lambda1=0,lambda2=self.lambda2,M=torch.inf, beta_tilde1=beta_tilde1,
                            beta_tilde2=beta_tilde2, L=self.L, iht_max_itr=self.num_iterations, ftol = 1e-7)
            elif self.algo == 'Heuristic_LS':
                w_pruned, obj, _, sol_time = torch_algo.Heuristic_LS(y,grads,w1,k,alpha=alpha_vec,lambda1=0,lambda2=self.lambda2,M=None, beta_tilde1=beta_tilde1, 
                        beta_tilde2=beta_tilde2, use_prune=True)
            elif self.algo == 'Heuristic_LSBlock':
                w_pruned, obj, _, sol_time = torch_algo.Heuristic_LSBlock(w1,grads,w1,k,alpha=alpha_vec,lambda1=0,lambda2=self.lambda2,M=None, beta_tilde1=beta_tilde1, 
                            beta_tilde2=beta_tilde2, use_prune=True,block_list=self.block_list, split_type=1)
            elif self.algo == 'IHT_LSBlock':
                w_pruned, obj, _, sol_time = torch_algo.IHT_LSBlock_PP(y,grads,w1,k,block_list=self.block_list,alpha=alpha_vec,lambda1=0,lambda2=self.lambda2,M=torch.inf, beta_tilde1=beta_tilde1, 
                            beta_tilde2=beta_tilde2,iht_max_itr=self.num_iterations)

            end_algo = time.time()
            self.update_model(w_pruned)

            #print('OPT took',end_algo-start_algo)

            #set_pvec(w_pruned, self.model,self.params,self.device)

            


        self.results['trace_C'] = (trace_C).cpu().item()
        self.results['trace_H']=(trace_H)
        self.results['gradient_norm']=(gradient_norm).cpu().item()
        self.results['norm_w_wbar']=(torch.linalg.norm(w_pruned-w1,ord=2)).cpu().item()
        self.results['sparsity']=(sparsity)
        new_nz = (w_pruned[w1 == 0] != 0).sum()
        self.results['new_non_zeros']=(new_nz).cpu().item()
        self.results['obj']=(obj).cpu().item()
        self.results['prun_runtime']=(end_algo - start_algo)
        self.results['norm_w']=(torch.linalg.norm(w_pruned,ord=2)).cpu().item()
        #self.results['test_acc']=(compute_acc(self.model,self.test_dataloader,self.device))


        new_mask = (w_pruned != 0).cpu()

        

        return w_pruned,new_mask
        
