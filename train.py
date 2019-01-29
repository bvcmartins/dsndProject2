#!/usr/bin/python3.6

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
import time
from PIL import Image
import argparse

'''
This program trains a convolutional NN for image recognition
'''

parser = argparse.ArgumentParser(
description = "This program trains a convolutional NN for image\
 recognition"
)

# parser options
parser.add_argument("--arch", action="store", \
type=str, dest='model_name', default='resnet50',\
choices=['resnet50','vgg16'],\
help="Choose pretrained model")

parser.add_argument("--epochs", action="store", \
type=int, dest='epochs', default='2',\
help="Define number of training epochs")

parser.add_argument("--data_dir", action="store", \
type=str, dest='data_dir', default='flowers',\
help="Choose data files root path")

parser.add_argument("--hidden", action="append", \
type=int, dest='n_hidden', default = [],\
help="Define number of hidden layers. \
Multiple layers defined by invoking flag multiple times.")

parser.add_argument("--save_dir", action="store", \
type=str, dest='save_dir', default='.',\
help="Choose path to save files")

parser.add_argument("--gpu", action="store", \
type=bool, dest='device', default=False,\
help="Choose if GPU is used")

parser.add_argument("--learn_rate", action="store", \
type=float, dest='learn_rate', default='0.003',\
help="Define learn rate for training algorithm")

args = parser.parse_args()

epochs = args.epochs
data_dir = args.data_dir
if args.n_hidden:
    hidden_units = args.n_hidden
else:
    hidden_units = [1024,512]
save_dir = args.save_dir
device = args.device
lr = args.learn_rate

print('model name: ', args.model_name)
print('number of epochs: ', epochs)
print('data path: ', data_dir)
print('hidden layers: ', hidden_units)
print('save path: ', save_dir)
print('gpu: ', device)

# define internal parameters
batch_size = 64
n_output = 102 # number of flower categories

class neuralNet(object):
    
    def prepare_data(self,datadir, batch_size):

        # define transforms
        transf_train = transforms.Compose(\
        [transforms.Resize(255),\
        transforms.RandomResizedCrop(224),\
        transforms.RandomRotation(30),\
        transforms.RandomHorizontalFlip(),\
        transforms.ToTensor(),\
        transforms.Normalize([0.485,0.456,0.406],\
        [0.229,0.224,0.225])])

        transf_test = transf_valid = transforms.Compose([\
        transforms.Resize(255),\
        transforms.CenterCrop(224),\
        transforms.ToTensor(),\
        transforms.Normalize([0.485,0.456,0.406],\
        [0.229, 0.224, 0.225])])

        # load datasets
        data_train = datasets.ImageFolder(root=datadir+'/train',\
        transform = transf_train)
        data_valid = datasets.ImageFolder(root=datadir+'/valid/',\
        transform = transf_valid)
        data_test = datasets.ImageFolder(root=datadir+'/test/',\
        transform = transf_test)

        # define dataloaders
        loader_train = torch.utils.data.DataLoader(data_train,\
        batch_size=batch_size, shuffle=True)
        loader_valid = torch.utils.data.DataLoader(data_valid,\
        batch_size=batch_size, shuffle=True)
        loader_test = torch.utils.data.DataLoader(data_test,\
        batch_size=batch_size, shuffle=True)
    
        return loader_train, loader_valid, loader_test



if __name__=='__main__':
    
    # load data 
    nn = neuralNet() 
    loader_train, loader_valid, loader_test = \
    nn.prepare_data(data_dir, batch_size)
