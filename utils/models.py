import torch
import torch.nn as nn
from typing import Optional


def CalculateOutSize(model, channels, samples):
    '''
    Calculate the output based on input size.
    model is from nn.Module and inputSize is a array.
    '''
    device = next(model.parameters()).device
    x = torch.rand(1, 1, channels, samples).to(device)
    out = model(x)
    return out.shape[-1]

def LoadModel(model_name, Chans, Samples, n_classes):
    if model_name == 'EEGNet':
        model = EEGNet(classes_num=n_classes, Chans=Chans,Samples=Samples)
    elif model_name == 'DeepConvNet':
        model = DeepConvNet(classes_num=n_classes, Chans=Chans, Samples=Samples)
    elif model_name == 'ShallowConvNet':
        model = ShallowConvNet(classes_num=n_classes, Chans=Chans, Samples=Samples)
    elif model_name == '1D_LSTM':
        model = CNN_LSTM(channels=Chans, n_classes=n_classes, time_points=Samples, 
                         hidden_size=128, num_layers=2)
    elif model_name == 'BrainprintNet':
        model = BrainprintNet(kernels=[7, 15, 31, 63, 127], fs=250, temporalLayer='LogVarLayer',
                               in_channels=Chans, nbands=12, num_classes=n_classes)
    elif model_name == 'MSNet':
        model = MSNet(kernels=[7, 15, 31, 63, 127], temporalLayer='LogVarLayer',
                       in_channels=Chans, num_classes=n_classes)
    else:
        raise 'No such model'
    return model

# two layers MLP Classifier
class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_size:int=50):
        super(Classifier, self).__init__()
        self.flatten = nn.Flatten()  
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.clf = nn.Sequential(self.flatten,self.fc1,self.relu,self.fc2)

    def forward(self, x):
        x = self.clf(x)
        return x


class EEGNet(nn.Module):
    """
    paras:
    samples: 采样点数量; kernel_size: 脑电信号中卷积核大小,对应原论文中的(1,64)卷积核;
    f1: 时域滤波器数量; f2: 点积滤波器数量; D: 空域滤波器数量;
    drop_out: we use p = 0.5 for within-subject classification and p = 0.25 for cross-subject classification
    """
    def __init__(self,
                 classes_num: int,
                 Chans: int, # 输入信号的通道维数，也即C
                 Samples: int, # 一般来说,不同任务也就前三个变量取不同值
                 kernel_size: int = 64,
                 f1: int = 8,
                 f2: int = 16,
                 D: int = 2,
                 drop_out: Optional[float] = 0.5):
        super(EEGNet, self).__init__()
        self.classes_num = classes_num
        self.in_channels = Chans
        self.samples = Samples
        self.kernel_size = kernel_size
        self.f1 = f1
        self.f2 = f2
        self.D = D
        self.drop_out = drop_out

        # time-conv2d,aggregate the temporal information
        # (1,C,T) --> (f1,C,T) ,上采样
        self.block1 = nn.Sequential(
            # four directions:left, right, up, bottom ;参数一般是默认(31,32,0,0)
            nn.ZeroPad2d((self.kernel_size // 2 - 1,
                          self.kernel_size - self.kernel_size // 2, 0,0)),  
            nn.Conv2d(in_channels = 1,
                      out_channels = self.f1,
                      kernel_size = (1,self.kernel_size), # 一般是默认值(1,64)
                      stride = 1,
                      bias = False ),# conv后若接norm层,bias设置为False,降低算力开销
            nn.BatchNorm2d(num_features = self.f1))  
    
        # DepthwiseConv2D,aggregate the spatial infomation
        # f1*C*T -conv--> (D*f1,1,T) -avgpool--> (D*f1,1,T//4) 
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = self.f1,
                      out_channels = self.f1 * self.D,
                      kernel_size = (self.in_channels,1),
                      groups = self.f1, # 分组卷积，聚合每个通道上信息
                      bias = False),
            nn.BatchNorm2d(num_features = self.f1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1,4)),
            nn.Dropout(self.drop_out))

        # depth-separable conv = Depthwise Convolution + Pointwise Convolution
        # point-conv for aggregate the info from all channels and change the num of channels
        # (D*f1,1,T//4) -seperableconv--> (f2,1,T//4) -avgpool--> (f2,1,T//32)
        self.block3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels = self.f1 * self.D, # depthwise-conv
                      out_channels = self.f1 * self.D,
                      kernel_size = (1, 16),
                      groups = self.f1 * self.D,
                      bias = False),
            nn.BatchNorm2d(num_features = self.f1 * self.D),
            nn.Conv2d(in_channels = self.f1 * self.D, # pointwise-conv
                      out_channels = self.f2,
                      kernel_size = (1, 1),
                      bias = False),
            nn.BatchNorm2d(num_features = self.f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop_out))
        
        self.clf = Classifier(input_size=self.f2 * (self.samples // (4 * 8)), output_size=self.classes_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x) 
        x = self.block2(x)
        x = self.block3(x) 
        x = x.view(x.size(0),-1)
        out = self.clf(x)
        return out
    
    # 网络输出预测熵
    def pred_ent(self,x):
        logits = self(x)
        lsm = nn.LogSoftmax(dim=-1)
        log_probs = lsm(logits)
        probs = torch.exp(log_probs)
        p_log_p = log_probs * probs
        predictive_entropy = -p_log_p.sum(axis=1)
        return predictive_entropy


class DeepConvNet(nn.Module):
    def __init__(self,
                 Chans: int,
                 Samples: int,
                 classes_num : int,
                 dropoutRate: Optional[float] = 0.5,
                 d1: Optional[int] = 25,
                 d2: Optional[int] = 50,
                 d3: Optional[int] = 100):
        super(DeepConvNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.classes_num = classes_num

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d1, kernel_size=(1, 5)),
            nn.Conv2d(in_channels=d1, out_channels=d1, kernel_size=(Chans, 1)),
            nn.BatchNorm2d(num_features=d1), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=d1, out_channels=d2, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=d2), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=d2, out_channels=d3, kernel_size=(1, 5)),
            nn.BatchNorm2d(num_features=d3), nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Dropout(self.dropoutRate))

        self.convT = CalculateOutSize(nn.Sequential(self.block1,self.block2,self.block3),self.Chans,self.Samples)
        self.clf = Classifier(input_size=self.convT*d3, output_size=self.classes_num) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = output.reshape(output.size(0), -1)
        out = self.clf(output)
        return out

    def MaxNormConstraint(self):
        for block in [self.block1, self.block2, self.block3]:
            for n, p in block.named_parameters():
                if hasattr(n, 'weight') and (
                        not n.__class__.__name__.startswith('BatchNorm')):
                    p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)


class Activation(nn.Module):
    def __init__(self, type):
        super(Activation, self).__init__()
        self.type = type

    def forward(self, input):
        if self.type == 'square':
            output = input * input
        elif self.type == 'log':
            output = torch.log(torch.clamp(input, min=1e-6))
        else:
            raise Exception('Invalid type !')

        return output


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        classes_num: int,
        Chans: int,
        Samples: int,
        dropoutRate: Optional[float] = 0.5, midDim: Optional[int] = 40,
    ):
        super(ShallowConvNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.classes_num = classes_num

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=midDim, kernel_size=(1, 13)),
            nn.Conv2d(in_channels=midDim,
                      out_channels=midDim,
                      kernel_size=(self.Chans, 1)),
            nn.BatchNorm2d(num_features=midDim), 
            nn.ELU(), #Activation('square'),
            nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7)),
            nn.ELU(), # Activation('log'), 
            nn.Dropout(self.dropoutRate))
        self.convT = CalculateOutSize(self.block1,self.Chans,self.Samples)
        self.clf = Classifier(input_size=self.convT*midDim, output_size=self.classes_num) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = output.reshape(output.size(0), -1)
        out = self.clf(output)
        return out

    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if hasattr(n, 'weight') and (
                    not n.__class__.__name__.startswith('BatchNorm')):
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=2.0)
                
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

"""
    Neural Network: CLSTM
    Detail: The input first cross CNN model ,then the output of CNN as the input of LSTM
"""


class CNN_LSTM(nn.Module):
    def __init__(self,channels,time_points,hidden_size,n_classes,num_layers,spatial_num=32,drop_out=0.25):
        super(CNN_LSTM, self).__init__()

        self.channels = channels
        self.time_points = time_points
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.drop_out = drop_out  
        self.spatial_num = spatial_num
        self.num_layers = num_layers

        self.block1 = nn.Sequential(
            nn.Conv2d(1,self.spatial_num,(self.channels,1),bias=False),
            nn.BatchNorm2d(self.spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(self.drop_out))
        self.block2 = nn.Sequential(
            nn.Conv2d(self.spatial_num,2*self.spatial_num,(1,1),bias=False),
            nn.BatchNorm2d(2*self.spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(self.drop_out))
        self.block3 = nn.Sequential(
            nn.Conv2d(2*self.spatial_num,4*self.spatial_num,(1,1),bias=False),
            nn.BatchNorm2d(4*self.spatial_num),
            nn.ELU(),
            nn.AvgPool2d((1, 2)),
            nn.Dropout(self.drop_out))
        self.convblock = nn.Sequential(self.block1,self.block2,self.block3)
        self.convT = CalculateOutSize(self.convblock,self.channels,self.time_points)
        
        self.lstm = nn.LSTM(8*self.convT, self.hidden_size, self.num_layers, batch_first=True)
        self.clf = nn.Sequential(nn.Linear(in_features=self.hidden_size*self.spatial_num//2, out_features=self.hidden_size),
                                 nn.Linear(in_features=self.hidden_size, out_features=self.n_classes))
        
    
    def forward(self,x:torch.Tensor)-> torch.Tensor:
        # X (batch_size,1,channels,time_points) = (B,1,C,T)
        x = self.block1(x) # (B,spatial_num,1,T//2)
        x = self.block2(x) # (B,2*spatial_num,1,T//4)
        x = self.block3(x) # (B,4*spatial_num,1,T//8)  eg:(32,128,1,125)
        x = x.reshape(x.shape[0],-1,8*self.convT) # (32,16,1000) # (B,spatial_num//2,T) 
        x, _ = self.lstm(x) # (B,spatial_num//2,hidden_size) eg:(32,16,192)
        x = x.reshape(x.shape[0],-1)
        return self.clf(x)

from data_loader import SubBandSplit
#%% Support classes for FBNet Implementation
class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim = self.dim, keepdim= True)

class StdLayer(nn.Module):
    '''
    The standard deviation layer: calculates the std of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(StdLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.std(dim = self.dim, keepdim=True)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))

class MeanLayer(nn.Module):
    '''
    The mean layer: calculates the mean of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim = self.dim, keepdim=True)

class MaxLayer(nn.Module):
    '''
    The max layer: calculates the max of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        ma ,ima = x.max(dim = self.dim, keepdim=True)
        return ma

temporal_layer = {'VarLayer': VarLayer, 'StdLayer': StdLayer, 'LogVarLayer': LogVarLayer, 'MeanLayer': MeanLayer, 'MaxLayer': MaxLayer}

class BrainprintNet(nn.Module):

    def __init__(self, kernels, fs, temporalLayer = 'LogVarLayer', strideFactor= 5,
                    in_channels:int=22, nbands=12, num_classes=9, radix=8):
        super(BrainprintNet, self).__init__()
        self.fs = fs
        self.kernels = kernels # type(List); conv_window kernels size
        self.parallel_conv = nn.ModuleList()
        self.strideFactor = strideFactor
        
        # 1D-parallel_conv
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv2d(in_channels=nbands, out_channels=nbands, kernel_size=(1,kernel_size),
                               stride=1, padding=0, bias=False, groups=nbands)
            self.parallel_conv.append(sep_conv)

        self.convblock = nn.Sequential(
            nn.BatchNorm2d(num_features=nbands),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=nbands, out_channels=nbands*radix, kernel_size=(in_channels,1),
                               stride=1, padding=0, bias=False)
            )
        
        self.temporalLayer = temporal_layer[temporalLayer](dim=-1)

        
        self.fc = nn.Sequential(
            nn.Linear(in_features=nbands*radix*strideFactor, out_features=num_classes),
            nn.LogSoftmax(dim=1)
        )
        
    
    def _get_fea_dim(self,x):
        with torch.no_grad():
            features, _ = self.forward(x)
        return features.shape

    def forward(self, x):
        # 为了在dataloader时统一expand_dims方便, 这里检查维度
        # x = x.squeeze(1) if x.dim() == 5 else x   # N, 1, nbands, C, T -> N, nbands, C, T
        x = torch.squeeze(x)
        device = x.device
        x = SubBandSplit(x.cpu().detach().numpy(), freq_start=8, freq_end=32, bandwidth=2, fs=self.fs)
        x = torch.from_numpy(x).to(device)
        out_sep = []
        # forward paralle 1D-conv blocks
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)
        out = torch.cat(out_sep, dim=-1)
        out = self.convblock(out)
        out = torch.squeeze(out) # N, C', T'

        pad_length = self.strideFactor - (out.shape[-1] % self.strideFactor)
        if pad_length != 0:
            out = F.pad(out, (0, pad_length))

        out = out.reshape([*out.shape[0:2], self.strideFactor, int(out.shape[-1]/self.strideFactor)])
        out = self.temporalLayer(out)
        out = torch.flatten(out, start_dim=1)

        features = out
        return self.fc(features)
    


class MSNet(nn.Module):

    def __init__(self, kernels, hidden_chans:int=64, temporalLayer = 'LogVarLayer', strideFactor= 5,
                    in_channels:int=22,  num_classes=9):
        super(MSNet, self).__init__()
        self.kernels = kernels # type(List); conv_window kernels size
        self.planes = hidden_chans # the channel num of all the hidden layers(中间所有隐藏层通道数)
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels
        self.strideFactor = strideFactor
        
        # 1D-parallel_conv
        for i, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.planes, kernel_size=(kernel_size),
                               stride=1, padding=0, bias=False,)
            self.parallel_conv.append(sep_conv)

        self.convblock = nn.Sequential(
            nn.BatchNorm1d(num_features=self.planes),
            nn.ReLU(inplace=False),
            nn.Conv1d(in_channels=self.planes, out_channels=self.planes, kernel_size=5,
                               stride=2, padding=2, bias=False),
            nn.BatchNorm1d(num_features=self.planes),
            nn.ReLU(inplace=False),
            )
        
        self.temporalLayer = temporal_layer[temporalLayer](dim=-1)

        # self.fc = nn.Linear(in_features=self.planes*strideFactor, out_features=num_classes)
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.planes*strideFactor, out_features=num_classes),
            nn.LogSoftmax(dim=1)
        )

    def _get_fea_dim(self,x):
        with torch.no_grad():
            features, _ = self.forward(x)
        return features.shape

    def forward(self, x):
        x = x.squeeze(1) if x.dim() == 4 else x  # N, 1, C, T -> N, C, T
        out_sep = []
        # forward paralle 1D-conv blocks
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)
        out = torch.cat(out_sep, dim=-1)
        out = self.convblock(out)

        pad_length = self.strideFactor - (out.shape[-1] % self.strideFactor)
        if pad_length != 0:
            out = F.pad(out, (0, pad_length))

        out = out.reshape([*out.shape[0:2], self.strideFactor, int(out.shape[-1]/self.strideFactor)])
        out = self.temporalLayer(out)
        out = torch.flatten(out, start_dim=1)

        features = out
        return self.fc(features)
    