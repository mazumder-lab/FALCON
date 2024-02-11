import string
from pruners.utils import *
from pruners.IHT_flop import IHTFLOPPruner
from pruners.multi_stage_pruner import MultiStagePruner
import json
import sys
import argparse
from itertools import product
import time
import os
from torch.utils.data import DataLoader
import copy
#MFACPATH = 'MFAC'
#sys.path.append(MFACPATH)
#from main_prun import  MagPruner,MFACPruner





parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='mlpnet')
parser.add_argument('--dset', type=str, default='mnist')

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--exp_id',type=str,default='')

parser.add_argument('--compute_training_losses', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--restrict_support', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--shuffle_train', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--use_wbar', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--use_activeset', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--test_batch_size', type=int, default=500)
parser.add_argument('--fisher_data_bsz', type=int, nargs='+')



parser.add_argument('--fisher_subsample_size', type=int, nargs='+')
parser.add_argument('--fisher_mini_bsz', type=int, nargs='+')

parser.add_argument('--num_iterations', type=int, nargs='+')
parser.add_argument('--num_stages', type=int, nargs='+')
parser.add_argument('--seed', type=int, nargs='+')
parser.add_argument('--first_order_term', type=lambda x: (str(x).lower() == 'true'), nargs='+')
parser.add_argument('--compute_trace_H', type=lambda x: (str(x).lower() == 'true'), nargs='+')
parser.add_argument('--alpha_one', type=lambda x: (str(x).lower() == 'true'), nargs='+')
parser.add_argument('--recompute_X', type=lambda x: (str(x).lower() == 'true'), nargs='+')
parser.add_argument('--sparsity', type=float, nargs='+')
parser.add_argument('--base_level', type=float, default=0.1) ##In correspondance with sparsity
parser.add_argument('--flop_ratio', type=float, nargs='+')
parser.add_argument('--flop_base', type=float, default=0.1) 
parser.add_argument('--l2', type=float, nargs='+')
parser.add_argument('--l2_logspace', type=lambda x: (str(x).lower() == 'true'), default=False) ##A different way to provide l2 list by giving 3 floats
parser.add_argument('--sparsity_schedule', type=str, nargs='+')
parser.add_argument('--algo', type=str, nargs='+')
parser.add_argument('--normalize', type=lambda x: (str(x).lower() == 'true'), nargs='+')
parser.add_argument('--block_size', type=int, nargs='+') ##Set to -1 if algo does not use this
parser.add_argument('--split_type', type=int, nargs='+') ##Set to -1 if algo does not use this
parser.add_argument('--alpha_scale', type=float, default=1)




args = parser.parse_args()
arch = args.arch
dset = args.dset
num_workers = args.num_workers
exp_name = args.exp_name
alpha_scale = args.alpha_scale

if len(args.fisher_data_bsz) > 1:
    fisher_sizes = [(args.fisher_subsample_size[i], args.fisher_mini_bsz[i], args.fisher_data_bsz[i]) for i in range(len(args.fisher_mini_bsz)) ] ##Glue the sizes together so we don't take products
else:
    fisher_sizes = [(args.fisher_subsample_size[i], args.fisher_mini_bsz[i], args.fisher_data_bsz[0]) for i in range(len(args.fisher_mini_bsz)) ] ##Glue the sizes together so we don't take products

if 'IMAGENET_PATH' in os.environ:
    IMAGENET_PATH = os.environ['IMAGENET_PATH'] +'/raw'
else:
    IMAGENET_PATH = '.'
dset_paths = {'imagenet':IMAGENET_PATH,'cifar10':'~/NetworkPruning/datasets',
                'mnist':'~/NetworkPruning/datasets'}

dset_path = dset_paths[dset]


acc_different_methods = []
X = None
base_level = args.base_level
flop_base = args.flop_base
ROOT_PATH = '.'

print("id is ",args.exp_id)
acc_different_methods = []
FOLDER = '{}/results/{}_{}_{}'.format(ROOT_PATH,arch,dset,exp_name)
FILE =  FOLDER+'/data{}_{}.csv'.format(args.exp_id,str(int(time.time())))
os.makedirs(FOLDER,exist_ok=True)

old_fisher_size,old_seed = None,None

if args.alpha_one is None:
    args.alpha_one  = [False]

if args.l2_logspace:
    if len(args.l2) != 3:
        raise ValueError('l2 arguments needs to provide exactly 3 values')
    args.l2 = np.logspace(args.l2[0],args.l2[1],int(args.l2[2]))
    print('l2 used',args.l2)

for seed,fisher_size, num_stages,num_iterations,first_order_term,compute_trace_H,recompute_X,sparsity,flop_ratio,l2,sparsity_schedule,algo,normalize,block_size,split_type,alpha_one in product(args.seed, fisher_sizes,args.num_stages, args.num_iterations,args.first_order_term,args.compute_trace_H,args.recompute_X,args.sparsity,args.flop_ratio,args.l2,args.sparsity_schedule,args.algo,args.normalize,args.block_size,args.split_type,args.alpha_one):
    
    print('seed,fisher_size, num_stages,num_iterations,first_order_term,compute_trace_H,recompute_X,sparsity,l2,sparsity_schedule,algo,normalize,block_size,split_type',seed,fisher_size, num_stages,num_iterations,first_order_term,compute_trace_H,recompute_X,sparsity,l2,sparsity_schedule,algo,normalize,block_size,split_type)

    if normalize and not algo == 'Heuristic_CD':
        continue
    if compute_trace_H and not first_order_term:
        continue 
    if alpha_one and not first_order_term:
        continue 
    if alpha_one and compute_trace_H:
        continue

    if  seed != old_seed or old_fisher_size != fisher_size:
        X = None

    if seed != old_seed:
        set_seed(seed)

        model,train_dataset,test_dataset,criterion,modules_to_prune = model_factory(arch,dset_path)

        train_dataloader = DataLoader(train_dataset, batch_size=args.test_batch_size, shuffle=True,num_workers=num_workers,pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True,num_workers=num_workers,pin_memory=True)

        ##########
        recompute_bn_stats = (arch == 'resnet20')
        if recompute_bn_stats:
            device='cpu'
            model.train()
            original_acc = compute_acc(model,train_dataloader,device=device) #sets up bn stats
            model.eval()
        ################
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using ', device)

        if torch.cuda.device_count()  > 1:
            print('Using DP with',torch.cuda.device_count(),'GPUs')
            model = torch.nn.DataParallel(model)
            modules_to_prune = ["module." + x for x in modules_to_prune]

        model.to(device)
        model.eval()
        start_test = time.time()
        dense_acc = compute_acc(model,test_dataloader,device=device)
        time_test = time.time() - start_test
        print('Dense test accuracy', dense_acc,' computation time : ', time_test)
        ###############

    if arch == 'resnet20':
        model.name = 'ResNetCifar'
    elif arch == 'mobilenetv1':
        model.name = 'MobileNet'
    elif arch == 'resnet50':
        model.name = 'ResNet'
    elif arch == 'resnet50down':
        model.name = 'ResNetdown'
    elif arch == 'WideResNet-28-10_CIFAR10':
        model.name = 'wideresnet'
    elif arch == 'WideResNet-28-10_CIFAR100':
        model.name = 'wideresnet2'
        
        
    if sparsity == -1:
        f_w = generate_weight(model.name)
        w_bar = get_pvec(model, modules_to_prune)
        idx_w = np.argsort(-np.abs(w_bar.to('cpu').numpy().astype(np.float64)))
        idx_s = np.searchsorted(np.cumsum(f_w[idx_w]), np.sum(f_w)* (1-flop_ratio))
        sparsity = 1-idx_s/len(w_bar)
        
    if sparsity == 0:
        
        f_w = generate_weight(model.name)
        w_bar = get_pvec(model, modules_to_prune)
        idx_w = np.argsort(-np.abs(w_bar.to('cpu').numpy().astype(np.float64) / (f_w)**(1/2)))
        idx_s = np.searchsorted(np.cumsum(f_w[idx_w]), np.sum(f_w)* (1-flop_ratio))
        sparsity = 1-idx_s/len(w_bar)
        

    fisher_subsample_size, fisher_mini_bsz,fisher_data_bsz = fisher_size
    prun_dataloader = DataLoader(train_dataset, batch_size=fisher_mini_bsz, shuffle=True,num_workers=num_workers,pin_memory=True)

    model_pruned = copy.deepcopy(model)
    
    pruner = IHTFLOPPruner(model_pruned,modules_to_prune,prun_dataloader,fisher_subsample_size,fisher_mini_bsz,criterion,block_size,l2,num_iterations,
        first_order_term,compute_trace_H,alpha_one, device,algo, alpha_scale)

    
    mask = torch.ones_like(get_pvec(model,modules_to_prune)).cpu() != 0
    multi_stage_pruner = MultiStagePruner(pruner,test_dataloader,sparsity_schedule,num_stages)

    start = time.time()
    w_pruned,mask = multi_stage_pruner.prune_flop(mask,sparsity,base_level,flop_ratio, flop_base,grads=X)
    end=time.time()
    del model_pruned
    acc_different_methods.append({'algo':algo,'sparsity':sparsity,'flop':flop_ratio,'l2':l2,'use_wbar':args.use_wbar,'use_activeset':args.use_activeset,'first_order_term':first_order_term,'runtime':end-start,'seed':seed,'fisher_mini_bsz':fisher_mini_bsz,'fisher_subsample_size':fisher_subsample_size,'fisher_data_bsz':fisher_data_bsz,'num_iterations':num_iterations,'num_stages':num_stages,'recompute_X':recompute_X,'shuffle_train' : args.shuffle_train,'recompute_bn_stats' : recompute_bn_stats,'ignore_bias':True,'compute_trace_H':compute_trace_H,'base_level':base_level,'flop_base':flop_base,'compute_training_losses':args.compute_training_losses,'restrict_support': args.restrict_support,'sparsity_schedule':sparsity_schedule,'normalize':normalize,'block_size':block_size,'split_type':split_type,'alpha_one':alpha_one, 'alpha_scale':alpha_scale})
    acc_different_methods[-1]['results'] = multi_stage_pruner.results
    with open(FILE, "w") as file:
        json.dump(acc_different_methods, file, cls=NpEncoder)

    old_fisher_size,old_seed = fisher_size,seed
    if X is None:
        X=multi_stage_pruner.pruner.grads


