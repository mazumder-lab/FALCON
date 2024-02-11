import string
from pruners.utils import *
from pruners.IHT_flop import IHTFLOPPruner
from pruners.multi_stage_pruner import MultiStagePruner
from pruners.gradual_pruner import GradualPruner
import json
import sys
import argparse
from itertools import product
import time
import os
from torch.utils.data import DataLoader
import copy
from utils.lr_schedules import cosine_lr_restarts,mfac_lr_schedule
MFACPATH = 'MFAC'
sys.path.append(MFACPATH)
from main_prun import  MagPruner,MFACPruner
import torch.distributed as dist
import builtins



parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='mlpnet')
parser.add_argument('--dset', type=str, default='mnist')
parser.add_argument('--training_schedule', type=str)

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--exp_id',type=str,default='')

parser.add_argument('--compute_training_losses', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--restrict_support', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--shuffle_train', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--use_wbar', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--use_activeset', type=lambda x: (str(x).lower() == 'true'), default=True)
parser.add_argument('--train_batch_size', type=int, default=500)
parser.add_argument('--test_batch_size', type=int, default=500)
parser.add_argument('--fisher_data_bsz', type=int, nargs='+')



parser.add_argument('--fisher_subsample_size', type=int, nargs='+')
parser.add_argument('--fisher_mini_bsz', type=int, nargs='+')

parser.add_argument('--num_iterations', type=int, nargs='+')
parser.add_argument('--num_stages', type=int, nargs='+')
parser.add_argument('--seed', type=int, nargs='+')
parser.add_argument('--first_order_term', type=lambda x: (str(x).lower() == 'true'), nargs='+')
parser.add_argument('--compute_trace_H', type=lambda x: (str(x).lower() == 'true'), nargs='+')
parser.add_argument('--recompute_X', type=lambda x: (str(x).lower() == 'true'), nargs='+')
parser.add_argument('--sparsity', type=float, nargs='+')
parser.add_argument('--base_level', type=float, default=0.1) ##In correspondance with sparsity
parser.add_argument('--flop_ratio', type=float, nargs='+')
parser.add_argument('--flop_base', type=float, default=0.1) 
parser.add_argument('--outer_base_level',type=float,default=0.5)
parser.add_argument('--l2', type=float, nargs='+')
parser.add_argument('--sparsity_schedule', type=str, nargs='+')
parser.add_argument('--algo', type=str, nargs='+')
parser.add_argument('--normalize', type=lambda x: (str(x).lower() == 'true'), nargs='+')
parser.add_argument('--block_size', type=int, nargs='+') ##Set to -1 if algo does not use this
parser.add_argument('--split_type', type=int, nargs='+') ##Set to -1 if algo does not use this

parser.add_argument('--max_lr',type=float)
parser.add_argument('--min_lr',type=float)
parser.add_argument('--ft_min_lr',type=float,default=-1)
parser.add_argument('--ft_max_lr',type=float,default=-1)
parser.add_argument('--prune_every',type=int)
parser.add_argument('--nprune_epochs',type=int)
parser.add_argument('--nepochs',type=int)
parser.add_argument('--gamma_ft',type=float,default=-1)
parser.add_argument('--warm_up', type=int, default=0)
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--first_epoch', type=int, default=0)
parser.add_argument('--schedule', type=str, default='cosine_lr_restarts')



args = parser.parse_args()
arch = args.arch
dset = args.dset
num_workers = args.num_workers
exp_name = args.exp_name
print(args.max_lr,args.nprune_epochs)


if len(args.fisher_data_bsz) > 1:
    fisher_sizes = [(args.fisher_subsample_size[i], args.fisher_mini_bsz[i], args.fisher_data_bsz[i]) for i in range(len(args.fisher_mini_bsz)) ] ##Glue the sizes together so we don't take products
else:
    fisher_sizes = [(args.fisher_subsample_size[i], args.fisher_mini_bsz[i], args.fisher_data_bsz[0]) for i in range(len(args.fisher_mini_bsz)) ] ##Glue the sizes together so we don't take products

if "WORLD_SIZE" in os.environ:
    args.world_size = int(os.environ["WORLD_SIZE"])
else:
    args.world_size = 1
args.distributed = args.world_size > 1

if args.distributed:
    if 'SLURM_PROCID' in os.environ: # for slurm scheduler
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
else:
    args.rank = 0
    args.gpu=0

#print(args.rank, 'start sleep')
#time.sleep(args.gpu*10)
#print(args.rank, 'end sleep')
exec(open('/usr/share/modules/init/python.py').read())
if True or args.gpu == 0:
    module('purge')
    module('load', 'anaconda/2022b')
    module('load', '/home/gridsan/groups/datasets/ImageNet/modulefile')

dset_paths = {'imagenet':os.environ['IMAGENET_PATH']+'/raw','cifar10':'~/NetworkPruning/datasets',
                'mnist':'~/NetworkPruning/datasets'}

dset_path = dset_paths[dset]

try:
    model,train_dataset,test_dataset,criterion,modules_to_prune = model_factory(arch,dset_path,pretrained=True)
except:
    module('load', '/home/gridsan/groups/datasets/ImageNet/modulefile')
    model,train_dataset,test_dataset,criterion,modules_to_prune = model_factory(arch,dset_path,pretrained=True)


###### Load from checkpoint
if len(args.checkpoint_path) >0:
    checkpoint = torch.load(args.checkpoint_path)
    new_state_trained = OrderedDict()
    for k in checkpoint['model_state_dict']:
        new_state_trained[k[7:]] = checkpoint['model_state_dict'][k]
    model.load_state_dict(new_state_trained)


####
if args.distributed:
    device = args.gpu
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device ', device)


ngpus_per_node = torch.cuda.device_count()

if args.distributed:
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.world_size, rank=args.rank)
    torch.backends.cudnn.benchmark = True

    if args.rank!=0:
        def print_pass(*args):
            pass
        #builtins.print = print_pass

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    model_without_ddp = model.module
    #modules_to_prune = ["module." + x for x in modules_to_prune]
    modules_to_prune = [x for x in modules_to_prune]
else:
    model = model.to(device)
    model_without_ddp = model

if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
else:
    train_sampler = None
train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=(train_sampler is None),num_workers=num_workers,pin_memory=True,sampler=train_sampler)
test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True,num_workers=num_workers,pin_memory=True)

alpha_one=False

##########
recompute_bn_stats = (arch == 'resnet20')
if recompute_bn_stats:
    device='cuda'
    model.train()
    original_acc = compute_acc(model,train_dataloader,device=device) #sets up bn stats
    model.eval()
################
#model.to(device)
model.eval()
if (not args.distributed or  args.rank == 0):
    #start_test = time.time()
    #dense_acc = compute_acc(model_without_ddp,test_dataloader,device=device)
    #time_test = time.time() - start_test
    #print('Dense test accuracy', dense_acc,' computation time : ', time_test)
    pass
###############

if arch == 'resnet20':
    model.name = 'ResNetCifar'
elif arch == 'mobilenetv1':
    model.name = 'MobileNet'
elif arch == 'resnet50':
    model.name = 'ResNet'

acc_different_methods = []
X = None
w_bar = None
base_level = args.base_level
flop_base = args.flop_base
ROOT_PATH = '/home/gridsan/groups/Sparse_NN'


acc_different_methods = []
FOLDER = '{}/results/{}_{}_{}'.format(ROOT_PATH,arch,dset,exp_name)
FILE =  FOLDER+'/data{}_{}.csv'.format(args.exp_id,str(int(time.time())))
os.makedirs(FOLDER,exist_ok=True)

old_fisher_size,old_seed = None,None

torch.backends.cudnn.benchmark = True

for seed,fisher_size, num_stages,num_iterations,first_order_term,compute_trace_H,recompute_X,sparsity,flop_ratio,l2,sparsity_schedule,algo,normalize,block_size,split_type in product(args.seed, fisher_sizes,args.num_stages, args.num_iterations,args.first_order_term,args.compute_trace_H,args.recompute_X,args.sparsity,args.flop_ratio,args.l2,args.sparsity_schedule,args.algo,args.normalize,args.block_size,args.split_type):
    
    if normalize and not algo == 'Heuristic_CD':
        continue
    if compute_trace_H and not first_order_term:
        continue 
    
    print('seed,fisher_size, num_stages,num_iterations,first_order_term,compute_trace_H,recompute_X,sparsity,l2,sparsity_schedule,algo,normalize,block_size,split_type',seed,fisher_size, num_stages,num_iterations,first_order_term,compute_trace_H,recompute_X,sparsity,l2,sparsity_schedule,algo,normalize,block_size,split_type)

    X = None
    nepochs = args.nepochs
    nprun_epochs = args.nprune_epochs
    reset_optimizer=True
    momentum = 0.9
    weight_decay = 0.00003751757813
    #sparsities = generate_schedule(nprun_epochs,args.base_level,sparsity,'poly')
    sparsities = generate_schedule(nprun_epochs,args.outer_base_level,sparsity,'poly')
    flop_ratios = generate_schedule(nprun_epochs,args.outer_base_level,flop_ratio,'poly')
    prun_every = args.prune_every
    gamma_ft = args.gamma_ft
    prunepochs = [(i) *prun_every for i in range(len(sparsities))]
    ignore_bias = True

    if args.ft_max_lr == -1:
        args.ft_max_lr = None
    if args.ft_min_lr == -1:
        args.ft_min_lr = None

    if args.schedule == 'mfac':
        lr_schedule = mfac_lr_schedule(nepochs,args.max_lr,args.min_lr,nprun_epochs,args.warm_up)
        print(lr_schedule)
    elif args.schedule == 'cosine_lr_restarts':
        lr_schedule = cosine_lr_restarts(nepochs,args.max_lr,args.min_lr,nprun_epochs,prun_every,gamma_ft,args.warm_up,ft_max_lr=args.ft_max_lr,ft_min_lr=args.ft_min_lr)
        print(args.ft_max_lr,args.ft_min_lr,'----<>')
    else:
        print('Unrecognized schedule')
        break

    if sparsity == -1:
        f_w = generate_weight(model.name)
        w_bar = get_pvec(model, modules_to_prune)
        idx_w = np.argsort(-np.abs(w_bar.to('cpu').numpy().astype(np.float64)))
        idx_s = np.searchsorted(np.cumsum(f_w[idx_w]), np.sum(f_w)* (1-flop_ratio))
        sparsity = 1-idx_s/len(w_bar)
        
    fisher_subsample_size, fisher_mini_bsz,fisher_data_bsz = fisher_size
    prun_dataloader = DataLoader(train_dataset, batch_size=fisher_mini_bsz, shuffle=True,pin_memory=True,num_workers=num_workers)
    acc_different_methods.append({'algo':algo,'l2':l2,'training_schedule':args.training_schedule,'max_lr':args.max_lr,'min_lr':args.min_lr,'nepochs':args.nepochs,'nprun_epochs':nprun_epochs,'prun_every':prun_every,'use_wbar':args.use_wbar,'use_activeset':args.use_activeset,'first_order_term':first_order_term,'seed':seed,'fisher_mini_bsz':fisher_mini_bsz,'fisher_subsample_size':fisher_subsample_size,'fisher_data_bsz':fisher_data_bsz,'num_iterations':num_iterations,'num_stages':num_stages,'recompute_X':recompute_X,'shuffle_train' : args.shuffle_train,'recompute_bn_stats' : recompute_bn_stats,'ignore_bias':True,'compute_trace_H':compute_trace_H,'base_level':base_level,'compute_training_losses':args.compute_training_losses,'restrict_support': args.restrict_support,'sparsity_schedule':sparsity_schedule,'normalize':normalize,'block_size':block_size,'split_type':split_type})
    
    model_pruned = model_without_ddp
   
    pruner = IHTFLOPPruner(model_pruned,modules_to_prune,prun_dataloader,fisher_subsample_size,fisher_mini_bsz,criterion,block_size,l2,num_iterations,
        first_order_term,compute_trace_H,alpha_one,
        device,algo)
    multi_stage_pruner = MultiStagePruner(pruner,test_dataloader,sparsity_schedule,num_stages)
    mask = get_pvec(model_without_ddp,modules_to_prune).cpu() != 0
    gradual_pruner = GradualPruner(multi_stage_pruner,train_dataloader,test_dataloader,criterion,
        modules_to_prune,reset_optimizer,momentum,weight_decay,acc_different_methods,FILE,seed,model=model,device=device,mask=mask,first_epoch=args.first_epoch,distributed=args.distributed,rank=args.rank,world_size=args.world_size)

    gradual_pruner.prune_flop(nepochs,lr_schedule,prunepochs,sparsities,flop_ratios)

    with open(FILE, "w") as file:
        json.dump(acc_different_methods, file,cls=NpEncoder)

    old_fisher_size,old_seed = fisher_size,seed


