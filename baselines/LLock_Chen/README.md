# Learnability-Lock
This is the official code for our paper [Learnability Lock: Authorized Learnability Control Through Adversarial Invertible Transformations](https://openreview.net/forum?id=6VpeS27viTq), accepted at ICLR 2022.

Refer to QuickStart notebook to generate unlearnable examples on CIFAR10.

### Prerequisites
 - Python 3.6
 - Pytorch
 - Numpy

### Usage
For i-Resnet block based lock, initialize using:
```
// customize the params following QuickStart
params = {'in_shape':32, 
               'n_channel':3, 
               'n_class':10,
               'mid_planes':32}
lock = iResLock(lock_params = params, epsilon=<>)
```
Linear transformation lock:
```
lock = LinearLock(lock_params = params, epsilon=<>)
```
