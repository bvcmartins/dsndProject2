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
This program trains a convolutional NN for image recognition using
pretrained model.
'''

#### Begin Parser configuration ####
parser = argparse.ArgumentParser(
description = "This program trains a convolutional NN for \
flower image recognition"
)

# Parser options

# Rubric: The training script allows users to choose from at least two 
# different architectures available from torchvision.models

# choose pretrained model [resnet50, vgg16]
parser.add_argument("--arch", action="store", \
type=str, dest='arch', default='resnet50',\
choices=['resnet50','vgg16'],\
help="Choose pretrained model")

# Rubric: The training script allows users to set hyperparameters for 
# learning rate, number of hidden units, and training epochs

# define number of epochs
parser.add_argument("--epochs", action="store", \
type=int, dest='epochs', default='2',\
help="Define number of training epochs")

# define root path for flowers directory
parser.add_argument("--data_dir", action="store", \
type=str, dest='data_dir', default='flowers',\
help="Choose root path for flowers directory")

# define number of nodes for the two hidden layers
parser.add_argument("--hidden", action="store", \
type=int, dest='n_hidden', nargs='+',\
help="Define number of nodes for each of the\
 two hidden layers (Example: train.py --hidden 10,10)")

# define path to save files
parser.add_argument("--save_dir", action="store", \
type=str, dest='save_dir', default='./',\
help="Define path to save files")

# Rubric: The training script allows users to choose 
# training the model on a GPU

# choose if GPU is used
parser.add_argument("--gpu", action="store", \
type=bool, dest='device', default=False,\
help="Choose if GPU is used")

# define learn rate
parser.add_argument("--learn_rate", action="store", \
type=float, dest='learn_rate', default='0.003',\
help="Define learn rate for training algorithm")

# Process parser options
args = parser.parse_args()

arch = args.arch
epochs = args.epochs
data_dir = args.data_dir
if args.n_hidden:
    hidden_units = args.n_hidden
else:
    hidden_units = [1024,512]
save_dir = args.save_dir

if args.device:
    device = 'cuda' 
else:
    device = 'cpu'
lr = args.learn_rate

# Print parser options
print('model name: ', arch)
print('number of epochs: ', epochs)
print('data path: ', data_dir)
print('hidden layers: ', hidden_units)
print('save path: ', save_dir)
print('gpu: ', device)

#### End parser configuration ####

# Define internal parameters
batch_size = 64
n_output = 102 # defined number of flower categories
# Open output log file
fh = open("output.dat","w")


def plotTraining(train_losses, valid_losses, acc):
    '''
    This function plots Epoch and Accuracy as a
    function of epoch for model training.
    '''
    
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

    '''
    This class generates, trains, and tests
    a convolutional neural network. The model
    is used for flower image recognition.
    '''
    
    def __init__(self, device):
        # define private variables
        self.__device = device 
        self.__optimizer = None
        self.__loss_crit = None
        self.__model = None
       

    def prepare_data(self, datadir, batch_size):
        '''
        This function reads the dataset (train, validation, test)
        and transforms it for processing by the model.
        '''
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
        '''
        This function reads the processed dataset and
        returns loaders (iterators) for the model.
        '''
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
        '''
        This function loads the chosen pretrained model.
        Default model is resnet50. It returns
        the number of inputs for the classifier
        according to the model.
        '''
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

    def classifier(self, n_inputs, n_hidden, n_output):
        '''
        This function builds the classifier that is coupled
        to the last layer of the pretained model.
        '''
        # define classifier operations
        clf = nn.Sequential(OrderedDict([\
                ('fc1', nn.Linear(n_inputs, n_hidden[0])),\
                ('relu1', nn.ReLU()),\
                ('dropout1', nn.Dropout(p=0.2)),\
                ('fc2', nn.Linear(n_hidden[0], n_hidden[1])),\
                ('relu2', nn.ReLU()),\
                ('dropout2', nn.Dropout(p=0.2)),\
                ('fc3', nn.Linear(n_hidden[1], n_output)),\
                ('log_softmax', nn.LogSoftmax(dim=1))]))
        
        # initialize weights using He initialization
        torch.nn.init.kaiming_normal_(clf.fc1.weight, \
                mode='fan_in')
        torch.nn.init.kaiming_normal_(clf.fc2.weight, \
                mode='fan_in')
        torch.nn.init.kaiming_normal_(clf.fc3.weight, \
                mode='fan_in')
        
        # attach classifier to last layer of pretrained model
        if arch == 'vgg16':
            self.__model.classifier = clf
        else: # defaults to resnet50
            self.__model.fc = clf

        return self.__model

    def valid_model(self, validloader):
        '''
        This function validates a training step
        using the validation set. It returns
        the validation loss and accuracy.
        '''
        # enter eval mode - turn off dropout
        self.__model.eval()
        # set return variables to zero
        valid_loss = 0
        accuracy = 0
        # iterate through validation set
        for inputs, labels in validloader:
            inputs, labels = inputs.to(self.__device), \
            labels.to(self.__device)
            # get logits from forward pass
            logit_v = self.__model.forward(inputs)
            # calculate loss from logits
            loss_v = self.__loss_crit(logit_v, labels)
            # accumulate losses for each batch to get total loss
            valid_loss += loss_v.item()
            # calculate probability using exp
            # to extract values from log-softmax
            prob = torch.exp(logit_v)
            # count how many predictions were right
            is_equal = (labels.data == prob.max(dim=1)[1])
            # get mean count for the batch step
            # accumulate counts for each batch to get total count
            accuracy += torch.mean(is_equal.type(torch.FloatTensor))
            # return model to train mode
            model.train()
        
        return valid_loss/len(validloader), accuracy/len(validloader)
    
    def train_model(self, epochs, \
            trainloader, validloader, lr=0.003):
        '''
        This function trains the model and returns
        training and validation losses and accuracy for each
        step.
        '''
        
        # using negative Log-Likelihood as loss function
        self.__loss_crit = nn.NLLLoss()

        # using Adam optimizer
        if arch == 'resnet50':
            self.__optimizer = \
                    optim.Adam(self.__model.\
                    fc.parameters(), lr)
        else:
            self.__optimizer = \
                    optim.Adam(self.__model.\
                    classifier.parameters(), lr)

        # define accumulation lists for plot
        train_losses = []
        valid_losses = []
        acc = []
        # define one validation step per batch
        valid_step = batch_size
        step_valid = 0
 
        # move model to device
        self.__model.to(self.__device)
        # start training mode - dropout activated
        self.__model.train()

        print(self.__model)

        for e in range(epochs):
            
            start = time.time()
            train_loss = 0
            steps = 0
            for inputs, labels in trainloader:
                inputs, labels = \
                            inputs.to(self.__device), \
                            labels.to(self.__device)     
                # set gradient to zero for each batch
                self.__optimizer.zero_grad()
                # get logits from forward pass
                logit = self.__model.forward(inputs)
                # calculate loss function with logits
                loss = self.__loss_crit(logit, labels)
                # perform backpropagation
                loss.backward()
                self.__optimizer.step()
                # accumulate train_loss
                train_loss += loss.item()
                steps += 1
                
                # enter validation part
                if steps % valid_step == 0:
                    valid_loss, accuracy = self.valid_model(validloader) 
                    model.train()

                # Rubric: The training loss, validation loss, and validation accuracy are 
                # printed out as a network trains
                    print("Epoch: {} of {}".format(e+1, epochs))
                    print("Training loss = {:.3f}".format(train_loss/len(trainloader)))
                    print("Validation loss = {:.3f}".format(valid_loss))
                    print("Validation Accuracy = {:.3f}".format(accuracy))
                    print(f"Time = {time.time()/60.:.3f}")
                    
                    fh.write("Epoch: {} of {}\n".format(e+1, epochs))
                    fh.write("Training loss = {:.3f}\n".format(train_loss/len(trainloader)))
                    fh.write("Validation loss = {:.3f}\n".format(valid_loss))
                    fh.write("Validation Accuracy = {:.3f}\n".format(accuracy))
                    step_valid += 1    
                    acc.append(accuracy / len(validloader))
                    valid_losses.append(valid_loss)
                    train_losses.append(train_loss)
        
        return [train_losses, valid_losses, acc]

                    
    
    def test_model(self, test_loader):

        '''
            This function loads the trained model and evaluates loss and 
            accuracy using the test set
        '''

        self.__model.to(device)
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            self.__model.eval()
            for inputs, labels in test_loader:
                inputs, labels = \
                        inputs.to(self.__device), \
                        labels.to(self.__device)
                log_prob = self.__model(inputs)
                loss = self.__loss_crit(log_prob, labels)
                test_loss += loss.item()
                prob = torch.exp(log_prob)
                #top_p, top_class = prob.topk(1, dim=1)
                #equals = top_class == labels.view(*top_class.shape)
                #accuracy += torch.mean(equals.type(torch.FloatTensor))
                is_equal = (labels.data == prob.max(dim=1)[1])
                accuracy += torch.mean(is_equal.type(torch.FloatTensor))
                    
        print("Test loss = {:.3f}".format(test_loss/len(test_loader)))
        print("Test accuracy = {:.3f}".format(accuracy/len(test_loader)))
        fh.write("Test loss = {:.3f}\n".format(test_loss/len(test_loader)))
        fh.write("Test accuracy = {:.3f}\n".format(accuracy/len(test_loader)))
         
        self.__model.train()
                
        return test_loss/len(test_loader), accuracy/len(test_loader)

    def save_model(self, n_inputs, n_hidden, n_output, checkpoint_path, data_train):
    
        '''
            This function saves the trained model
        '''
        
        self.__model.class_to_idx = data_train.class_to_idx
        checkpoint = {\
                'input_size': n_inputs, \
                'hidden_size': n_hidden, \
                'output_size': n_output, \
                'conv_model': arch, \
                'class_to_idx': self.__model.class_to_idx, \
                'state_dict': self.__model.state_dict()}
    
        torch.save(checkpoint, checkpoint_path+'checkpoint_final.pth')

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

    # replace classifier of pretrained network
    model = cnn.classifier(n_inputs, hidden_units, n_output)

    # Rubric: train.py successfully trains a new network on a dataset of images and 
    # saves the model to a checkpoint
    # train the network
    train_losses, valid_losses, acc = cnn.train_model(epochs, \
            loader_train, loader_valid, lr)

    #plotTraining(train_losses, valid_losses, acc)

    # test the model
    test_loss, accuracy = cnn.test_model(loader_test)

    # save the model 
    cnn.save_model(n_inputs, hidden_units, n_output, save_dir, data_train)
    
    fh.close()