from errno import ENETUNREACH
import numpy as np
import numpy.linalg as la
import numba as nb
from time import time
from sklearn.utils import extmath
from collections import namedtuple
import warnings
from numba.core.errors import NumbaDeprecationWarning, \
    NumbaPendingDeprecationWarning, NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
from numba import prange
import torch
import torch.nn as nn

def get_macs_dpf(input_res, device, model, multiply_adds=False, ignore_zero=True, display_log=True,
                 ignore_bn=False, ignore_relu=False, ignore_maxpool=False, ignore_bias=False):


    # Code from https://github.com/simochen/model-tools.
    '''
    if "cifar" in dset:
        input_res = [3, 32, 32]
    elif "imagenet" in dset:
        input_res = [3, 224, 224]
    elif "mnist" in dset:
        input_res = [1, 28, 28]
    '''
    
    
    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2["names"] = np.prod(input[0].shape)

    list_conv = []
    module_names = []

    def conv_hook(self, input, output):
        # print(self.weight.shape)
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = (
                self.kernel_size[0] * self.kernel_size[1] *
                (self.in_channels / self.groups)
        )
        bias_ops = 1 if not ignore_bias and self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        num_weight_params = (
            (self.weight.data != 0).float().sum()
            if ignore_zero
            else self.weight.data.nelement()
        )
        print(input_channels, output_channels, kernel_ops)
        print(num_weight_params,output_height,output_width,batch_size)
        assert self.weight.numel() == kernel_ops * output_channels, "Not match"
        flops = (
                (
                        num_weight_params * (2 if multiply_adds else 1)
                        + bias_ops * output_channels
                )
                * output_height
                * output_width
                * batch_size
        )

        list_conv.append(flops)
        module_names.append(self.name)

    list_linear = []


    def linear_hook(self, input, output):
        # print(self.weight.shape)
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        num_weight_params = (
            (self.weight.data != 0).float().sum()
            if ignore_zero
            else self.weight.data.nelement()
        )
        weight_ops = num_weight_params * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement() if not ignore_bias else 0

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
        module_names.append(self.name)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (
                (kernel_ops + bias_ops)
                * output_channels
                * output_height
                * output_width
                * batch_size
        )

        list_pooling.append(flops)

    list_upsample = []

    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)


    def foo(net, name=''):

        children = list(net.named_children())
        if not children:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
                setattr(net, 'name', name)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
                setattr(net, 'name', name)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(
                    net, torch.nn.AvgPool2d
            ):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for child_name, child in children:
            foo(child, name="{}.{}".format(name, child_name))

            
    assert model is not None
    # print(model)
    foo(model)
    _input = torch.rand(*input_res).unsqueeze(0).to(device)
    model(_input)

    total_flops = (
            sum(list_conv)
            + sum(list_linear)
            + (sum(list_bn) if not ignore_bn else 0)
            + (sum(list_relu) if not ignore_relu else 0)
            + (sum(list_pooling) if not ignore_maxpool else 0)
            + sum(list_upsample)
    )
    total_flops = (
        total_flops.item() if isinstance(total_flops, torch.Tensor) else total_flops
    )
    list_conv = [x.item() for x in list_conv]
    list_linear = [x.item() for x in list_linear]
    print("list conv is ", list_conv)
    print("list linear is ", list_linear)
    print("list bn is ", list_bn)
    print("list pool is ", list_pooling)
    print("list relu is ", list_relu)
    print("list module_names is ", module_names)

    print(sum(list_linear) + sum(list_conv))
    if display_log:
        print(
            "  + Number of {}: {:.3f}M".format(
                "flop" if multiply_adds else "macs", 1.0 * total_flops / 1e6
            )
        )
    return total_flops, list_conv + list_linear, module_names


def dp_minflop(w_tilde, neuron_list, weight_list, lambda2):
    
    w_pruned = np.copy(w_tilde)
    flop_list = []
    cur_idx = 0
    idx_list = [0]
    layer_list = []
    arg_list = []
    neuronbest_list = [neuron_list[-1]]
    num_layer = len(neuron_list)-1
    for i in range(num_layer):
        num_neuron = neuron_list[i] * neuron_list[i+1]
        layer_list.append(np.copy(w_tilde[cur_idx:cur_idx+num_neuron]))
        cur_idx += num_neuron
        idx_list.append(cur_idx)
    
    
    dp_dict = {}
    opt_dict = {}
    for i in range(num_layer):
        
        if i == 0:
            layer_weight = layer_list[i].reshape(neuron_list[i+1], neuron_list[i])
            layer_sum = np.sum(layer_weight**2,axis=0)
            arg_idx = np.argsort( -layer_sum )
            arg_list.append(arg_idx)
            layer_loss = np.sum(layer_sum) - np.cumsum(layer_sum[arg_idx])
            for j in range(neuron_list[i]+1):
                dp_dict[i,j] = layer_loss[j-1]
                
        else:
            layer_weight = layer_list[i].reshape(neuron_list[i+1], neuron_list[i])
            layer_sum = np.sum(layer_weight**2,axis=0)
            arg_idx = np.argsort( -layer_sum )
            arg_list.append(arg_idx)
            layer_loss = np.sum(layer_sum) - np.cumsum(layer_sum[arg_idx])
            for j in range(1,neuron_list[i]+1):
                dp_valist = [dp_dict[i-1,j2] + lambda2 * weight_list[i] * j2 * j  for j2 in range(1,neuron_list[i-1]+1)]
                arg_neuron = np.argmin(dp_valist)
                dp_dict[i,j] = layer_loss[j-1] + dp_valist[arg_neuron]
                opt_dict[i,j] = arg_neuron+1
    
    dp_finlist = [dp_dict[num_layer-1,j2] + lambda2 * weight_list[-1] * j2 * neuron_list[-1]  for j2 in range(1,neuron_list[-2]+1)]
    arg_neuron = np.argmin(dp_finlist)
    dp_dict[num_layer,neuron_list[-1]] = dp_finlist[arg_neuron]
    opt_dict[num_layer,neuron_list[-1]] = arg_neuron+1
    arg_list.append(np.arange(neuron_list[-1]))
    
    neuron_best = neuron_list[-1]
    for i in range(num_layer-1,-1,-1):
        
        layer_weight = layer_list[i].reshape(neuron_list[i+1], neuron_list[i])
        neuron_prev = opt_dict[i+1,neuron_best]
        
        arg_idx, arg_idx2 = arg_list[i], arg_list[i+1]
        layer_weight[:,arg_idx[neuron_prev:]] *= 0
        layer_weight[arg_idx2[neuron_best:],:] *= 0
        w_pruned[idx_list[i]:idx_list[i+1]] =  layer_weight.reshape(-1)
        flop_list.insert(0, np.sum(w_pruned[idx_list[i]:idx_list[i+1]] != 0))
        
        neuronbest_list.insert(0,neuron_prev)
        neuron_best = neuron_prev
    
    return w_pruned, flop_list, neuronbest_list


def flopcalc_mlpnet(w):
    
    neuron_list = [784,40,20,10]
    
    neuronact_list = []
    flop_list = []
    cur_idx = 0
    fc_list = []

    for i in range(len(neuron_list)-1):
        num_neuron = neuron_list[i] * neuron_list[i+1]
        fc_list.append(np.copy(w[cur_idx:cur_idx+num_neuron]))
        cur_idx += num_neuron
    
    neuronact_list.append(np.ones(neuron_list[-1])) 
    for i in range(len(neuron_list)-2,-1,-1):
        fc_weight = fc_list[i].reshape(neuron_list[i+1], neuron_list[i])
        fc_actweight = (fc_weight.T * neuronact_list[0]).T
        flop_list.insert(0, np.sum(fc_actweight != 0))
        neuronact_list.insert(0, (np.sum(fc_actweight != 0, axis=0) != 0).astype("float64"))
        
    return np.sum(flop_list), flop_list, neuronact_list