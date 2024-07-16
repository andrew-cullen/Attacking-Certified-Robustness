import torch

from typing import *
from datasets import dataset_load

import torchvision.utils
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

class SequentialNormalize(torch.nn.Module):
    def __init__(self, model, mean, std, cuda=True):
        super(SequentialNormalize, self).__init__()
        
        self.means = torch.tensor(mean)
        self.sds = torch.tensor(std)
        self.model = model

    def forward(self, input: torch.tensor):
        # The following repeatedly perturbs the mean and std - the repeated calculations end up being faster than pre-calculating and sub-sampling.
        (batch_size, _, height, width) = input.shape
        means = self.means.to(input.device).repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.to(input.device).repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)        
        x = (input - means) / sds        
        return self.model(x)
        
from torchvision.transforms.functional import normalize
                             
                                     
def model_settings(dataset: str, args):                                
    if dataset == 'mnist':
        return _mnist(args)
    elif dataset == 'cifar10':
        return _cifar10(args)
    elif dataset == 'tinyimagenet':
        return _tinyimagenet(args)
    elif dataset == 'imagenet':
        return _imagenet(args)


def _mnist(args):
    if args.batch_size == 0:
        batch_size = 128
    else: 
        batch_size = args.batch_size
        
    if args.lr == 0:
        lr = 0.001
    else:
        lr = args.lr
        
    train_loader  = torch.utils.data.DataLoader(dataset=dataset_load('mnist', 'train'),
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=10)

    val_loader = torch.utils.data.DataLoader(dataset=dataset_load('mnist', 'test'),
                                             batch_size=batch_size, 
                                             shuffle=True, num_workers=10)

    test_loader = torch.utils.data.DataLoader(dataset=dataset_load('mnist', 'test'),
                                             batch_size=1,
                                             shuffle=True, num_workers=8)

    model = models.resnet18(num_classes=10)
    model = SequentialNormalize(model, [0.1307], [0.3081])   
    
    cuda_device_count = torch.cuda.device_count()
    if args.parallel  == 'always':
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(cuda_device_count)])    
        device = torch.device("cpu")
    else: 
        device = torch.device("cuda" if cuda_device_count > 0 else "cpu")        
        model = model.to(device)    

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return model, loss, optimizer, None, train_loader, val_loader, test_loader, device, 10
    
def _cifar10(args):
    if args.batch_size == 0:
        batch_size = 150
    else: 
        batch_size = args.batch_size
        
    if args.lr == 0:
        lr = 0.001
    else:
        lr = args.lr   

    train_loader = torch.utils.data.DataLoader(dataset=dataset_load('cifar10', 'train'),
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=10)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_load('cifar10', 'test'),
                                              batch_size=1,
                                              shuffle=True, num_workers=10)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_load('cifar10', 'test'),
                                              batch_size=batch_size,
                                              shuffle=True, num_workers = 8)
                                   
    model = models.resnet18(num_classes=10)
    model = SequentialNormalize(model, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    
    cuda_device_count = torch.cuda.device_count()
    if args.parallel  == 'always':
        model = torch.nn.DataParallel(model, device_ids=[i for i  in range(cuda_device_count)])    
        device = torch.device("cpu")
    else:   
        device = torch.device("cuda" if cuda_device_count > 0 else "cpu")        
        model = model.to(device)  
    
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)                                           
        
    return model, loss, optimizer, None, train_loader, val_loader, test_loader, device, 10
    
def _tinyimagenet(args):
    if args.batch_size == 0:
        batch_size = 128
    else: 
        batch_size = args.batch_size
        
    if args.lr == 0:
        lr = 0.1
    else:    
        lr = args.lr

    train_loader = torch.utils.data.DataLoader(dataset=dataset_load('tinyimagenet', 'train'),
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=10)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_load('tinyimagenet', 'test'),
                                              batch_size=1,
                                              shuffle=True, num_workers=10)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_load('tinyimagenet', 'test'),
                                              batch_size=batch_size,
                                              shuffle=True, num_workers = 8)
    
    classes = 200
    
    model = models.resnet18(num_classes=classes)
    model.conv1 = torch.nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    model.fc.out_features = classes
    model = SequentialNormalize(model, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    cuda_device_count = torch.cuda.device_count()
    if args.parallel  == 'always':   
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(cuda_device_count)])    
        device = torch.device("cpu")
    else: 
        device = torch.device("cuda" if cuda_device_count > 0 else "cpu")        
        model = model.to(device) 

    
    loss = nn.CrossEntropyLoss()    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)#001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    return model, loss, optimizer, lr_scheduler, train_loader, val_loader, test_loader, device, 200
    
def _imagenet(args):
    if args.batch_size == 0:
        batch_size = 512 #256
    else: 
        batch_size = args.batch_size
        
    if args.lr == 0:
        lr = 0.1
    else:    
        lr = args.lr

    train_loader = torch.utils.data.DataLoader(dataset=dataset_load('imagenet', 'train'),
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=14)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_load('imagenet', 'test'),
                                              batch_size=1,
                                              shuffle=True, num_workers=14)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_load('imagenet', 'test'),
                                              batch_size=batch_size,
                                              shuffle=True, num_workers = 14)
   
    classes = 1000
    
    base_model = models.resnet18(num_classes=classes)
    base_model.conv1 = torch.nn.Conv2d(3,64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    base_model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    base_model.fc.out_features = classes
    model = SequentialNormalize(base_model, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    device = None
    
    loss = nn.CrossEntropyLoss()    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    return model, loss, optimizer, lr_scheduler, train_loader, val_loader, test_loader, device, classes   
            


