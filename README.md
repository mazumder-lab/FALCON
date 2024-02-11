# FALCON: FLOP-Aware Combinatorial Optimization for Neural Network Pruning

This is the offical repo of the AISTATS 2024 paper **FALCON: FLOP-Aware Combinatorial Optimization for Neural Network Pruning**

## Requirements
This code has been tested with Python 3.8 and the following packages:
```
numpy==1.22.0
torch==1.12.1+cu113
torchvision==0.13.1+cu113
numpy==0.56.4
```

Currently, we requires the Cifar10 and Imagenet datasets to prune the networks. Please replace the dataset paths in `run_expflop.py` and `run_expflop_gradual.py` with the corresponding local paths to these datasets.

## Structure of the repo
Scripts to run the algorithms are located in `scripts/`. The current code supports applying FALCON++ to prune the following architectures (datasets):  ResNet20 (Cifar10), MobileNetV1 (Imagenet) and ResNet50 (Imagenet). Results will be saved in `results/`. Adding new models can be done through `model_factory` function in `pruners/utils.py`. 


## Citing FALCON

