import os

from typing import *
import torchvision.datasets as dsets
import torchvision.transforms as transforms

IMAGENET_LOC_ENV = ... # "/data/gpfs/datasets/Imagenet/ILSVRC/Data/CLS-LOC"

DATASETS = ["mnist", "cifar10", "tinyimagenet", "imagenet"]


def dataset_load(dataset: str, split: str):
    if dataset == 'mnist':
        return _mnist(split)
    elif dataset == 'cifar10':
        return _cifar10(split)
    elif dataset == 'tinyimagenet':
        return _tinyimagenet(split)
    elif dataset == 'imagenet':
        return _imagenet(split)
    else:
       print('Incompatible dataset type')
       
def _imagenet(split: str):
    if split == "train":
        subdir = os.path.join(IMAGENET_LOC_ENV, "train") #_blurred") 
        transform = transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]) # transforms.RandomResizedCrop(224, scale=(0.2, 1.)), 
    elif split == "test":
        subdir = os.path.join(IMAGENET_LOC_ENV, "val") #"val_blurred")        
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])     
    return dsets.ImageFolder(subdir, transform) 
            
def _tinyimagenet(split: str):              
    raise ValueError('Tinyimagenet Loc currently not specified')

def _mnist(split: str):            
    if split == "train":    
        return dsets.MNIST(root='./data/',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)  
    elif split == "test":                            
        return dsets.MNIST(root='./data/',
                             train=False,
                             transform=transforms.ToTensor(),
                             download=True)
                                 
def _cifar10(split: str):
    if split == "train":
        return dsets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transforms.ToTensor())
    elif split == "test":
        return dsets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transforms.ToTensor())
