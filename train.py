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
type=str, dest='arch', default='resnet50',\
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

arch = args.arch
epochs = args.epochs
data_dir = args.data_dir
if args.n_hidden:
    hidden_units = args.n_hidden
else:
    hidden_units = [1024,512]
save_dir = args.save_dir
if args.device == True:
    device = 'cuda' 
else:
    device = 'cpu'
lr = args.learn_rate

print('model name: ', arch)
print('number of epochs: ', epochs)
print('data path: ', data_dir)
print('hidden layers: ', hidden_units)
print('save path: ', save_dir)
print('gpu: ', device)

# define internal parameters
batch_size = 64
n_output = 102 # number of flower categories

def plotTraining(train_losses, valid_losses, acc):
    fig, axes = plt.subplots(figsize=(12,4), ncols=2)
    axes[0].plot(train_losses, label='train')
    axes[0].plot(valid_losses, label='valid')
    axes[0].legend(loc='upper left')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[1].plot(acc)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')

    return fig

class convNeuralNet(object):

    def __init__(self, device):
        self.__device = device 
        self.__optimizer = None
        self.__loss_crit = None
        self.__model = None

    def prepare_data(self,datadir, batch_size):

        print('prepare_data')
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

        return data_train, data_valid, data_test

    def prepare_loader(self, data_train, data_valid, data_test, \
    batch_size):

        print('prepare_loader')
        # define dataloaders
        loader_train = torch.utils.data.DataLoader(data_train,\
        batch_size=batch_size, shuffle=True)
        loader_valid = torch.utils.data.DataLoader(data_valid,\
        batch_size=batch_size, shuffle=True)
        loader_test = torch.utils.data.DataLoader(data_test,\
        batch_size=batch_size, shuffle=True)
    
        return loader_train, loader_valid, loader_test

    def load_model(self, arch):
        if arch == 'vgg16':
            print('vgg16 loaded')
            self.__model = models.vgg16(pretrained=True)
            n_inputs = 25088
        else: # defaults to resnet50
            print('resnet50 loaded')
            self.__model = models.resnet50(pretrained=True)
            n_inputs = 2048
    
        # freeze feature parameters - do not backpropagate them
        for param in self.__model.parameters():
            param.requires_grad = False

        return n_inputs

    def update_clf(self, clf):
        if arch == 'vgg16':
            self.__model.classifier = clf
        else: # defaults to resnet50
            self.__model.fc = clf

        return self.__model

    def setup_training(self, lr=0.003):
#        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # using negative Log-Likelihood as loss function
        self.__loss_crit = nn.NLLLoss()

        # using Adam optimizer
        if arch == 'resnet50':
            self.__optimizer = \
                    optim.Adam(self.__model.fc.parameters(), lr)
        else:
            self.__optimizer = \
                    optim.Adam(self.__model.classifier.parameters(), lr)

        return self.__device

    def train_model(self, epochs, trainloader, validloader):
        train_losses = []
        valid_losses = []
        acc = []
        valid_step = batch_size
        step_valid = 0
 
        self.__model.to(self.__device)
        self.__model.train()


        print(self.__model)

        for e in range(epochs):
            print('epoch: ',e)
            start = time.time()
            train_loss = 0
            steps = 0
            for inputs, labels in trainloader:
                inputs, labels = \
                            inputs.to(self.__device), \
                            labels.to(self.__device)        
                self.__optimizer.zero_grad()
                logit = self.__model.forward(inputs)
                loss = self.__loss_crit(logit, labels)
                loss.backward()
                self.__optimizer.step()
                train_loss += loss.item()
                steps += 1
            
                if steps % valid_step == 0:
        
                    valid_loss = 0
                    accuracy = 0
                    with torch.no_grad():
                        self.__model.eval()
                        for inputs, labels in validloader:
                            inputs, labels = \
                                        inputs.to(self.__device), \
                                        labels.to(self.__device)
                            logit_v = self.__model.forward(inputs)
                            loss_v = self.__loss_crit(logit_v, labels)
                            valid_loss += loss_v.item()
                            prob = torch.exp(logit_v)
                            is_equal = (labels.data == prob.max(dim=1)[1])
                            accuracy += torch.mean(is_equal.type(torch.FloatTensor))
            
                    step_valid += 1    
                    model.train()

                    print("Epoch: {} of {}".format(e+1, epochs))
                    print("Training loss = {:.3f}".format(train_loss/len(trainloader)))
                    print("Validation loss = {:.3f}".format(valid_loss/len(validloader)))
                    print("Validation Accuracy = {:.3f}".format(accuracy/len(validloader)))
                    print(f"Time = {time.time()/60.:.3f}")
                    acc.append(accuracy / len(validloader))
                    valid_losses.append(valid_loss / len(validloader))
                    train_losses.append(train_loss / len(trainloader))
        
        return [train_losses, valid_losses, acc]

        def test_model(self, test_loader):

            ''''
            This function loads the trained model and evaluates its accuracy
            using the test set
            
            INPUTS:
            test_loader - Data loader containing test data
            
            OUTPUTS:
            test loss - mean loss calculated for testing set
            accuracy - mean accuracy calculated using the testing label
            '''
            
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                self.__model.eval()
                for inputs, labels in test_loader:
                    inputs, labels = \
                            inputs.to(self.__device), \
                            labels.to(self.__device)
                    log_prob = self.__model(inputs)
                    loss = self.loss_crit(log_prob, labels)
                    test_loss += loss.item()
                    prob = torch.exp(log_prob)
                    #top_p, top_class = prob.topk(1, dim=1)
                    #equals = top_class == labels.view(*top_class.shape)
                    #accuracy += torch.mean(equals.type(torch.FloatTensor))
                    is_equal = (labels.data == prob.max(dim=1)[1])
                    accuracy += torch.mean(is_equal.type(torch.FloatTensor))
                    
#                print("Test loss = {:.3f}".format(test_loss/len(test_loader)))
#                print("Test accuracy = {:.3f}".format(accuracy/len(test_loader)))
                
                self.__model.train()
                
            return test_loss, accuracy




class Clf(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = nn.Linear(hidden_size[1], output_size)
        
        self.dropout = nn.Dropout(p=0.2)
        # initialize weights using He initialization
        torch.nn.init.kaiming_normal_(self.fc1.weight, \
                mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc2.weight, \
                mode='fan_in')
        torch.nn.init.kaiming_normal_(self.fc3.weight, \
                mode='fan_in')

 
    def forward(self, x):

        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x

if __name__=='__main__':
    
    # instantiate neural network class  
    cnn = convNeuralNet(device) 

    # load and prepare data
    data_train, data_valid, data_test = \
    cnn.prepare_data(data_dir, batch_size)

    # prepare loaders 
    loader_train, loader_valid, loader_test = \
    cnn.prepare_loader(data_train, data_test, \
    data_valid, batch_size)
    n_inputs = cnn.load_model(arch)

    # instantiate classifier
    clf = Clf(n_inputs, hidden_units, n_output)

    # replace classifier of pretrained network
    model = cnn.update_clf(clf)
#    print(model)

    # setup training functions - device, loss and optimizer
    device = cnn.setup_training(lr)
    print('device: ',device)

    # train the network
    train_losses, valid_losses, acc = cnn.train_model(epochs, loader_train, loader_valid)

    #plotTraining(train_losses, valid_losses, acc)

    # test the model
    test_loss, accuracy = test_model(loader_test)

