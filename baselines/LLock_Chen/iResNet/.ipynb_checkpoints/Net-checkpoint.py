"""
Code taken and adapted from
"""

import torch.nn.functional as F
# import toolbox
import numpy as np
import torch
import torchvision
from torch import nn
from .model_utils import injective_pad, ActNorm2D, Split
from .model_utils import squeeze as Squeeze
from .model_utils import MaxMinGroup
from .spectral_norm_conv_inplace import spectral_norm_conv
from .spectral_norm_fc import spectral_norm_fc
from .matrix_utils import exact_matrix_logarithm_trace, power_series_matrix_logarithm_trace
from torch.distributions import constraints

class conv_iresnet_block(nn.Module):
    def __init__(self, in_shape, int_ch, numTraceSamples=0, numSeriesTerms=0,
                 stride=1, coeff=.97, input_nonlin=True,
                 actnorm=False, n_power_iter=5, nonlin="elu", epsilon=8/255):
        """
        buid invertible bottleneck block
        :param in_shape: shape of the input (channels, height, width)
        :param int_ch: dimension of intermediate layers
        :param stride: 1 if no downsample 2 if downsample
        :param coeff: desired lipschitz constant
        :param input_nonlin: if true applies a nonlinearity on the input
        :param actnorm: if true uses actnorm like GLOW
        :param n_power_iter: number of iterations for spectral normalization
        :param nonlin: the nonlinearity to use
        """
        super(conv_iresnet_block, self).__init__()
        assert stride in (1, 2)
        self.stride = stride
        self.squeeze = Squeeze(stride)
        self.coeff = coeff
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter
        self.epsilon = epsilon
        nonlin = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "softplus": nn.Softplus,
            "sorting": lambda: MaxMinGroup(group_size=2, axis=1)
        }[nonlin]

        # set shapes for spectral norm conv
        in_ch, h, w = in_shape
            
        layers = []
        if input_nonlin:
            layers.append(nonlin())

        in_ch = in_ch * stride**2
        kernel_size1 = 3 # kernel size for first conv
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(in_ch, int_ch, kernel_size=kernel_size1, stride=1, padding=1),
                                                  (in_ch, h, w), kernel_size1))
        layers.append(nonlin())
        kernel_size2 = 1 # kernel size for second conv
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(int_ch, int_ch, kernel_size=kernel_size2, padding=0),
                                                  (int_ch, h, w), kernel_size2))
        layers.append(nonlin())
        kernel_size3 = 3 # kernel size for third conv
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(int_ch, in_ch, kernel_size=kernel_size3, padding=1),
                                                  (int_ch, h, w), kernel_size3))
        self.bottleneck_block = nn.Sequential(*layers)
        if actnorm:
            self.actnorm = ActNorm2D(in_ch)
        else:
            self.actnorm = None
        

    def forward(self, x, ignore_logdet=False):
        """ bijective or injective block forward """
        if self.stride == 2:
            x = self.squeeze.forward(x)

        if self.actnorm is not None:
            x, an_logdet = self.actnorm(x)
        else:
            an_logdet = 0.0

        Fx = self.bottleneck_block(x)
        # apply tanh to constrain the output to (-epsilon, epsilon)
        Fx = self.epsilon * torch.tanh(Fx)
        
        # Compute approximate trace for use in training
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)

        # add residual to output
        y = Fx + x
        return y, x

    def inverse(self, y, maxIter=100):
        # inversion of ResNet-block (fixed-point iteration)
        x = y
        for iter_index in range(maxIter):
            summand = self.bottleneck_block(x)
            summand = self.epsilon * torch.tanh(summand)
            x = y - summand

        if self.actnorm is not None:
            x = self.actnorm.inverse(x)

        # inversion of squeeze (dimension shuffle)
        if self.stride == 2:
            x = self.squeeze.inverse(x)
        return x
    
    def _wrapper_spectral_norm(self, layer, shapes, kernel_size):
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            return spectral_norm_fc(layer, self.coeff, 
                                    n_power_iterations=self.n_power_iter)
        else:
            # use spectral norm based on conv, because bound not tight
            return spectral_norm_conv(layer, self.coeff, shapes,
                                      n_power_iterations=self.n_power_iter)

        
class conv_iresnet_block3(nn.Module):
    def __init__(self, in_shape, int_ch, numTraceSamples=0, numSeriesTerms=0,
                 stride=1, coeff=.97, input_nonlin=True,
                 actnorm=False, n_power_iter=5, nonlin="elu", epsilon=8/255):
        """
        buid invertible bottleneck block
        :param in_shape: shape of the input (channels, height, width)
        :param int_ch: dimension of intermediate layers
        :param stride: 1 if no downsample 2 if downsample
        :param coeff: desired lipschitz constant
        :param input_nonlin: if true applies a nonlinearity on the input
        :param actnorm: if true uses actnorm like GLOW
        :param n_power_iter: number of iterations for spectral normalization
        :param nonlin: the nonlinearity to use
        """
        super(conv_iresnet_block3, self).__init__()
        assert stride in (1, 2)
        self.stride = stride
        self.squeeze = Squeeze(stride)
        self.coeff = coeff
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter
        self.epsilon = epsilon
        nonlin = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "softplus": nn.Softplus,
            "sorting": lambda: MaxMinGroup(group_size=2, axis=1)
        }[nonlin]

        # set shapes for spectral norm conv
        in_ch, h, w = in_shape
            
        layers = []
#         if input_nonlin:
#             layers.append(nonlin())

        in_ch = in_ch * stride**2
        kernel_size1 = 3 # kernel size for first conv
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(in_ch, 32, kernel_size=kernel_size1, stride=1, padding=1),
                                                  (in_ch, h, w), kernel_size1))
        layers.append(nonlin())
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(32, 32, kernel_size=kernel_size1, stride=1, padding=1),
                                                  (32, h, w), kernel_size1))
        layers.append(nonlin())

        kernel_size2 = 1 # kernel size for second conv

        layers.append(self._wrapper_spectral_norm(nn.Conv2d(32, 32, kernel_size=kernel_size2, padding=0),
                                                  (128, h, w), kernel_size2))
        layers.append(nonlin())
        kernel_size3 = 3 # kernel size for third conv

        layers.append(self._wrapper_spectral_norm(nn.Conv2d(32, 8, kernel_size=kernel_size3, padding=1),
                                                  (32, h, w), kernel_size3))
        layers.append(nonlin())
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(8, in_ch, kernel_size=kernel_size3, padding=1),
                                                  (8, h, w), kernel_size3))
        self.bottleneck_block = nn.Sequential(*layers)
        self.layers = layers
        if actnorm:
            self.actnorm = ActNorm2D(in_ch)
        else:
            self.actnorm = None
    def forward(self, x, ignore_logdet=False):
        """ bijective or injective block forward """
        if self.stride == 2:
            x = self.squeeze.forward(x)

        if self.actnorm is not None:
            x, an_logdet = self.actnorm(x)
        else:
            an_logdet = 0.0

#         Fx = self.bottleneck_block(x)
        Fx = x
        for layer in self.layers:
            Fx = layer(Fx)
#         Fx = torch.clamp(Fx, -self.epsilon, self.epsilon)
        Fx= self.epsilon * torch.tanh(Fx)
        # Compute approximate trace for use in training
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)

        # add residual to output
        y = Fx + x
        return y, x

    def inverse(self, y, maxIter=100):
        # inversion of ResNet-block (fixed-point iteration)
        x = y
        for iter_index in range(maxIter):
            summand = self.bottleneck_block(x)
            summand = self.epsilon * torch.tanh(summand)
            x = y - summand

        if self.actnorm is not None:
            x = self.actnorm.inverse(x)

        # inversion of squeeze (dimension shuffle)
        if self.stride == 2:
            x = self.squeeze.inverse(x)
        return x
    
    def _wrapper_spectral_norm(self, layer, shapes, kernel_size):
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            return spectral_norm_fc(layer, self.coeff, 
                                    n_power_iterations=self.n_power_iter)
        else:
            # use spectral norm based on conv, because bound not tight
            return spectral_norm_conv(layer, self.coeff, shapes,
                                      n_power_iterations=self.n_power_iter)

class Conv_iResNet(nn.Module):
    def __init__(self, mid_planes=8, in_planes=3, in_shape = 32, num_classes=10, num_layers=1, epsilon=0.032):
        super(Conv_iResNet, self).__init__()
        self.blocklist = nn.ModuleList()
        self.epsilon = epsilon
        for i in range(num_layers):
            self.blocklist.append(conv_iresnet_block([in_planes, in_shape, in_shape], mid_planes, stride=1, actnorm=False, nonlin='relu', epsilon=self.epsilon/num_layers))

    def forward(self, x):
        out = x
        for block in self.blocklist:
            out, _ = block(out)
        return out
    
    def inverse(self, y, i=5):
        x_re = y
        for block in self.blocklist[::-1]:
            x_re = block.inverse(x_re, i)
        return x_re
