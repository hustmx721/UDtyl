import utils
import torch
import torch.nn as nn
from torch.autograd import Variable
from abc import ABC, abstractmethod
from trainer import *
from tqdm import tqdm
import os
import numpy as np
import iResNet


class Lock(nn.Module):
    """ Skeleton class to hold learnability lock.
        For general input-agnostic adversarial perturbations, see the
        ThreatModel class
        All subclasses need the following:
        - perturbation_norm() : no args -> scalar Variable
        - self.parameters() needs to iterate over params we want to optimize
        - constrain_params() : no args -> no return,
             modifies the parameters such that this is still a valid image
        - forward : no args -> Variable - applies the adversarial perturbation
                    the originals and outputs a Variable of how we got there
        - adversarial_tensors() : applies the adversarial transform to the
                                  originals and outputs TENSORS that are the
                                  adversarial images
    """
    def __init__(self, epsilon = 8/255, lock_params = None, device='cuda'):

        super(Lock, self).__init__()
        self.initialized = False
        self.lock_params = lock_params
        self.epsilon = epsilon
        self.device = device


    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.lock(x)
    
    """
        print images, perturbed images, and noise in 3 rows
    """
    def show_noise(self, x, y):
        pass
    
    @abstractmethod
    def train(self, loader):
        pass

    @abstractmethod
    def lock(self, x):
        pass

    @abstractmethod
    def unlock(self, x):
        pass

    @abstractmethod
    def save():
        pass

class LinearLock(Lock):
    """ Lock that contains a linear pixel-wise map for each class
        For general input-agnostic adversarial perturbations, see the
        ThreatModel class
    """
    def __init__(self, epsilon = 8/255, lock_params = {'in_shape':32, 'n_channel':3, 'n_class':10},
                 device='cuda'):

        super(LinearLock, self).__init__(epsilon, lock_params, device)
        self.n_class = self.lock_params['n_class']
        self.n_channel = self.lock_params['n_channel']
        self.input_shape = self.lock_params.get('input_shape')
        self.setup()  # initialize W and b
    
    
    def __call__(self, x):
        return self.forward(x)
    

    def setup(self, W=None, B=None):
        if self.initialized: print("Overwriting the weights...")
        img_shape = self.lock_params.get('in_shape')
        input_shape = self.input_shape

        if input_shape is not None:
            if isinstance(input_shape, int):
                input_shape = (input_shape,)
            weight_shape = (self.n_class, self.n_channel, *input_shape)
        else:
            weight_shape = (self.n_class, self.n_channel, img_shape, img_shape)

        if W is not None:
            self.W = torch.tensor(W, device=self.device)
        else:
            self.W = torch.ones(weight_shape, device=self.device)
            self.W = Variable(self.W, requires_grad=True)
        if B is not None:
            self.B = torch.tensor(B, device=self.device)
        else:
            self.B = torch.zeros(weight_shape, device=self.device)
            self.B = Variable(self.B, requires_grad=True)
        self.initialized = True
        
    
    def forward(self, x):
        return self.lock(x)
    
    
    def train(self, base_model, loader, opt_base = None, loss_base = None, learning_rate=0.01, I=30, J=2):
        optimizer, criterion = opt_base, loss_base
        if not optimizer:
            optimizer =  torch.optim.SGD(params=base_model.parameters(), lr=0.01, momentum=0.9)
        if not criterion:
            criterion = torch.nn.CrossEntropyLoss()
            
#         sdlr = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
        
        data_iter = iter(loader)
        it = 0
        
        flag = True
        while flag:
            # optimize theta for I steps
            base_model.train()
            self.W, self.B = Variable(self.W, requires_grad=False), Variable(self.B, requires_grad=False)
            for param in base_model.parameters():
                param.requires_grad = True
            for j in range(0, I):
                try:
                    (images, labels) = next(data_iter)
                except:
                    data_iter = iter(loader)
                    (images, labels) = next(data_iter)

                images, labels = images.to(self.device), labels.to(self.device)
                images = self.lock(images, labels).to(self.device)
                base_model.zero_grad()
                optimizer.zero_grad()
                logits = base_model(images)
                loss = criterion(logits, labels)
                loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 5.0)
                optimizer.step()   

            # Eval stop condition
            eval_idx, total, correct = 0, 0, 0
            base_model.eval()
            self.W, self.B = Variable(self.W, requires_grad=False), Variable(self.B, requires_grad=False)
            for i, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                images = self.lock(images, labels).to(self.device)
                with torch.no_grad():
                    logits = base_model(images)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = correct / total
            print('Accuracy %.2f' % (acc*100))
            
            # Perturbation over entire dataset
            idx = 0
            for param in base_model.parameters():
                param.requires_grad = False
            base_model.eval()
            self.W, self.B = Variable(self.W, requires_grad=True), Variable(self.B, requires_grad=True)
            ############## Using SGD by default: change if you want  ##################
            opt = torch.optim.SGD([self.W, self.B], lr=learning_rate, momentum=0.9)
            ###########################################################################
            for _ in range(J):
                for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
                    images, labels = images.to(self.device), labels.to(self.device)
                    perturb_img = self.lock(images, labels).to(self.device)
            #             perturb_img = torch.clamp(perturb_img, 0, 1).to(self.device)
                    opt.zero_grad()
                    base_model.zero_grad()
                    logits = base_model(perturb_img)
                    loss = criterion(logits, labels)
#                     loss = -1* loss
                    loss.backward(retain_graph=False)
                    opt.step()

#                     print(self.B)

#             sdlr.step()
            
            # Project into the constraint
            self.W = torch.clamp(self.W, 1-self.epsilon/2, 1+self.epsilon/2)
            self.B = torch.clamp(self.B, -self.epsilon/2, self.epsilon/2)


            # Eval stop condition
            eval_idx, total, correct = 0, 0, 0
            base_model.eval()
            self.W, self.B = Variable(self.W, requires_grad=False), Variable(self.B, requires_grad=False)
            for i, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                images = self.lock(images, labels).to(self.device)
                with torch.no_grad():
                    logits = base_model(images)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = correct / total
            print('Accuracy %.2f' % (acc*100))
            it = it + 1 # count iterations
            if acc > 0.90 or it>50:
                flag=False      
    
        
    
    def transform_sample(self, x, label):
        # element-wise/pixel-wise transform
        result = self.W[label] * x + self.B[label]  
#         result = result * 1/(1+self.epsilon)
        return result
    
    def inv_transform_sample(self, x, label):
#         x = x * (1 + self.epsilon)
        result = (x - self.B[label]) * 1/self.W[label]
        return result
    
    def lock(self, X, y):
        result = torch.zeros_like(X, device=self.device)
        for i in range(X.shape[0]):
            result[i] = self.transform_sample(X[i], y[i])
        return result

    def unlock(self, X, y):
        result = torch.zeros_like(X, device=self.device)
        for i in range(X.shape[0]):
            result[i] = self.inv_transform_sample(X[i], y[i])
        return result
    
    def get_params(self):
        W, b = None, None
        if self.device=='cuda':
            W, b = self.W.cpu().detach().numpy(), self.B.cpu().detach().numpy()
        else:
            W, b = self.W.detach().numpy(), self.B.detach().numpy()
        return W, b
    
    def save(self, sname='default', path='.'):
        W, b = self.get_params()
        np.save(os.path.join(path,'W_'+sname+'.npy'), W)
        np.save(os.path.join(path,'B_'+sname+'.npy'), b)
        print('Weights saved as {}'.format(os.path.join(path,sname)))
        
        
        


class iResLock(Lock):
    """ Lock that contains a linear pixel-wise map for each class
        For general input-agnostic adversarial perturbations, see the
        ThreatModel class
    """
    def __init__(self, epsilon = 8/255, lock_params = {'in_shape':32, 'n_channel':3, 'n_class':10, 'mid_planes':8},
                 device='cuda', sname = 'iResLock_default'):

        super(iResLock, self).__init__(epsilon, lock_params, device)
        self.n_class = self.lock_params['n_class']
        self.n_channel = self.lock_params['n_channel']
        self.sname = sname
        try:
            self.mid_planes = self.lock_params['mid_planes']
        except:
            print("Warning: missing mid_planes value - taking 16 !!!")
            self.mid_planes = 16
        self.setup()  # initialize W and b
    
    
    def __call__(self, x):
        return self.forward(x)
    
    def read_weights(self, sname, path):
        weights = list()
        if path is not None:
            p = os.path.join(path, sname)
            for i in range(self.n_class):
                w = torch.load(p + '_' + str(i)+'.pkl')
                weights.append(w)
        if len(weights) > 0: print("reading success!")
        return weights
    
    def setup(self, weights = None):
        if self.initialized: print("Overwriting the weights...")
        img_shape = self.lock_params['in_shape']
        self.transforms = list()
        for i in range(self.n_class):
            net = iResNet.Conv_iResNet( mid_planes=self.mid_planes, in_planes=self.n_channel, in_shape = img_shape, num_classes=self.n_class,
                                                          num_layers=1, epsilon=self.epsilon ).to(self.device)
            if weights is not None: net.load_state_dict(weights[i])
            self.transforms.append(net)
        
        self.initialized = True
        
    
    def forward(self, x):
        return self.lock(x)
    
    """
        Helper function - switch between training mode and evaluation mode
    """
    def switch_mode(self, mode='train'):
        if mode == 'train':
            for net in self.transforms:
                net.train()
                for param in net.parameters():
                    param.requires_grad = True 
        else:        
            for net in self.transforms:
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False 
        return True

    
    def train(self, base_model, loader, opt_base = None, loss_base = None, scheduler_base = None, learning_rate=0.001, I=30, J=2):
        optimizer, criterion, scheduler = opt_base, loss_base, scheduler_base
        if not optimizer:
            optimizer =  torch.optim.SGD(params=base_model.parameters(), lr=0.01, momentum=0.9)
        if not criterion:
            criterion = torch.nn.CrossEntropyLoss()
        if not scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.001)
        data_iter = iter(loader)
        it = 0
        
        flag = True
        while flag:
            # optimize theta for I steps
            base_model.train()
            for param in base_model.parameters():
                param.requires_grad = True
            self.switch_mode('eval')
            for j in range(0, I):
                try:
                    (images, labels) = next(data_iter)
                except:
                    data_iter = iter(loader)
                    (images, labels) = next(data_iter)

                images, labels = images.to(self.device), labels.to(self.device)
                images = self.lock(images, labels).to(self.device)
                base_model.zero_grad()
                optimizer.zero_grad()
                logits = base_model(images)
                loss = criterion(logits, labels)
                loss.backward(retain_graph=False)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), 5.0)
                optimizer.step()   
            scheduler.step()
            # Eval stop condition
            eval_idx, total, correct = 0, 0, 0
            for param in base_model.parameters():
                param.requires_grad = False
            base_model.eval()
            for i, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                images = self.lock(images, labels).to(self.device)
                with torch.no_grad():
                    logits = base_model(images)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = correct / total
            print('Accuracy %.2f' % (acc*100))
            
            # Optimize over the parameters of perturbation function for J steps
            idx = 0
            self.switch_mode('train')
            ############## Using SGD by default: change if necessary ##################
            opt = torch.optim.SGD(self.all_parameters(), lr=learning_rate, momentum=0.9)
            ###########################################################################
            for _ in range(J):
                for i, (images, labels) in tqdm(enumerate(loader), total=len(loader)):
                    images, labels = images.to(self.device), labels.to(self.device)
                    perturb_img = self.lock(images, labels).to(self.device)
            #             perturb_img = torch.clamp(perturb_img, 0, 1).to(self.device)
                    opt.zero_grad()
                    base_model.zero_grad()
                    logits = base_model(perturb_img)
                    loss = criterion(logits, labels)
#                     loss = -1* loss
                    loss.backward(retain_graph=False)
                    opt.step()


            # Eval stop condition
            eval_idx, total, correct = 0, 0, 0
            base_model.eval()
            
            
            
            self.switch_mode('eval')
            for i, (images, labels) in enumerate(loader):
                images, labels = images.to(self.device), labels.to(self.device)
                images = self.lock(images, labels).to(self.device)
                with torch.no_grad():
                    logits = base_model(images)
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            acc = correct / total
            print('Accuracy %.2f' % (acc*100))
            it = it + 1 # count iterations
            if acc > 0.90 or it>50:
                flag=False      
    
        
    
    def transform_sample(self, x, label):
        result = self.transforms[label].forward(torch.unsqueeze(x, 0))
#         result = (result - (-self.epsilon)) / ( 1 + 2 * self.epsilon )
        return torch.squeeze(result)
    
    def inv_transform_sample(self, x, label):
#         x = x * (1 + 2 * self.epsilon ) - self.epsilon
        result = self.transforms[label].inverse(torch.unsqueeze(x, 0))
        return torch.squeeze(result)
    
    def lock(self, X, y):
        result = torch.zeros_like(X, device=self.device)
        for i in range(X.shape[0]):
            result[i] = self.transform_sample(X[i], y[i])
        return result

    def unlock(self, X, y):
        result = torch.zeros_like(X, device=self.device)
        for i in range(X.shape[0]):
            result[i] = self.inv_transform_sample(X[i], y[i])
        return result
    
    ''' for passing into optimizers'''
    def all_parameters(self):
        result = []
        for net in self.transforms:
            result = result + list(net.parameters())
        return result
    
    """ for saving """
    def get_params(self):
        assert self.initialized, "Not properly initilized!!!!"
        return self.transforms
    
    def save(self, sname='default', path='.'):
        tlist = self.get_params()
        for i,t in enumerate(tlist):
            torch.save(t.state_dict(), os.path.join(path, sname + '_' + str(i) + '.pkl'))
        print('Weights saved as {}'.format(os.path.join(path,sname)))