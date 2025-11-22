
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import pickle
from utils import same_seeds, standard_loss, cifar10_training_data, add_random_gaussian_noise 
import time
import warnings
import argparse
import numpy as np
import csv
from torch.autograd import Variable 
from classifiers.resnet import resnet18, resnet50
from classifiers.vgg import VGG16, VGG19
from classifiers.densenet import DenseNet121
from classifiers.mobilenetv2 import MobileNetV2


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from sklearn.utils import check_random_state
    check_random_state(seed)

def COIN_trans(image, severity=4):
    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    alpha = [0.8, 1.2, 1.6, 2, 2.4, 2.8, 3.2][severity - 1]

    dx = (np.random.uniform(-alpha, alpha, size=shape[:2])).astype(np.float32)
    dy = (np.random.uniform(-alpha, alpha, size=shape[:2])).astype(np.float32)
 
    if len(image.shape) < 3 or image.shape[2] < 3:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    else:
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]),np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx,(-1, 1)), np.reshape(z, (-1, 1))
    trans_img = np.clip(map_coordinates(image, indices, order=1, mode='wrap').reshape(shape), 0, 1) * 255
    return trans_img




warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Testing the effectiveness of COIN')
parser.add_argument('--lr', default=0.1, type=float, help='learning-rate')
parser.add_argument('--epochs', default=80, type=int, help='number of epoch') 
parser.add_argument('--arch', default='resnet18', type=str, help='types of training architecture')   
parser.add_argument('--poison', default="CUDA", type=str, help="CUDA, VUDA, HUDA, ...")
parser.add_argument('--batch_size', default=128, type=int)  
parser.add_argument('--seed', default=0, type=int)    
parser.add_argument('--severity', default=4, type=int, help="severity=4 denotes alpha=2.0") 
parser.add_argument('--coin', action='store_true', help="whether to employ COIN defense")
args = parser.parse_args() 


same_seeds(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




if args.coin:         #applying COIN
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: np.uint8(COIN_trans(x, severity=args.severity))),
        transforms.ToTensor(),
    ]) 
 

else:        #applying standard transformation
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
 

 
training_data_path = os.path.join("./poisoned_data/cifar10", args.poison + '.pkl')     #poisoned training data
 
train_dataset = cifar10_training_data(training_data_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True) 

test_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

 

if args.dataset == 'cifar10':
    num_classes = 10
elif args.dataset == 'cifar100':
    num_classes = 100
if args.arch == "resnet18":
    net = resnet18(in_dims=3, out_dims=num_classes)
elif args.arch == "resnet50":
    net = resnet50(in_dims=3, out_dims=num_classes)
elif args.arch == "vgg16":
    net = VGG16(num_classes=num_classes)
elif args.arch == "vgg19":
    net = VGG19(num_classes=num_classes)
elif args.arch == "densenet121":
    net = DenseNet121(num_classes=num_classes)
elif args.arch == 'mobilenetv2':
    net = MobileNetV2(num_classes=num_classes)
 
net = net.to(device) 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

for epoch in range(args.epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    net.train()
    for i, (inputs, labels) in enumerate(train_loader, 0):      #index starts from 0
        inputs = torch.clamp(inputs, 0, 1)
        labels = labels.long()
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss, _ = standard_loss(args, net, inputs, labels)
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        loss.backward()
        optimizer.step()

    print('[Epoch：%d/%d] loss: %.3f Train Acc: %.3f' % (epoch + 1, args.epochs, running_loss / len(train_loader), 100. * correct / total)) 
    running_loss = 0.0

    if (epoch + 1) % 5 == 0: 
        net.eval()
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(test_loader, 0):
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        print('Test Acc: %.2f' % (100. * correct / total)) 

 
with open(os.path.join(f'results.csv'), 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow([args.coin, args.arch, args.poison, 100 * correct / total])  
print('Finished Training')


 


























 



 

