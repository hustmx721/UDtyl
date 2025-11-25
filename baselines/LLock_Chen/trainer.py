from tqdm import tqdm
from .utils import read_config
import torch
import torch.nn as nn 


torch_config = read_config("config.ini")['TORCH']
use_cuda = torch_config['gpu']
device = torch.device("cuda" if use_cuda else "cpu")


"""
 code adapted from tutorial https://adversarial-ml-tutorial.org/adversarial_training/
 One epoch regular training/evaluation
"""
def epoch(loader, model, opt=None, loss_f = None, device = "cuda"):
    total_loss, total_err = 0.,0.
    pbar = tqdm(loader, total=len(loader))
    for X,y in pbar:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        if not loss_f:    
            loss_f = nn.CrossEntropyLoss()
        loss = loss_f(yp, y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()    
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

"""Adversarial training/evaluation epoch over the dataset"""
def epoch_adversarial(loader, model, attack, opt=None, **kwargs):

    total_loss, total_err = 0.,0.
    pbar = tqdm(loader, total=len(loader))
    for X,y in pbar:
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y, **kwargs)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


"""
    One batch without logging;
"""
def batch(X, y, model, opt, loss_f = None, device = "cuda"):
    X,y = X.to(device), y.to(device)
    yp = model(X)
    if not loss_f:    
        loss_f = nn.CrossEntropyLoss()
    loss = loss_f(yp, y)
    if opt:
        opt.zero_grad()
        loss.backward()
        opt.step()    
    return True 


"""
    One batch returning error rate and loss;
"""
def batch_(X, y, model, opt, loss_f = None, device = "cuda"):
    total_loss, total_err = 0.,0.
    X,y = X.to(device), y.to(device)
    yp = model(X)
    if not loss_f:    
        loss_f = nn.CrossEntropyLoss()
    loss = loss_f(yp, y)
    if opt:
        opt.zero_grad()
        loss.backward()
        opt.step()    
    total_err += (yp.max(dim=1)[1] != y).sum().item()
    total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)



def fgsm(model, X, y, epsilon=8/255):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd_linf(model, X, y, epsilon=0.0312, alpha=0.006, num_iter=10, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()