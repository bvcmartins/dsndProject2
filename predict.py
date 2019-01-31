#!/usr/bin/python

import numpy as np
#import pandas as pd
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

# define input variables - testing
path_image = "./"
checkpoint_path = "./"
top_k = 5
category_names = 'cat_to_name.json'
gpu_test = False

'''

This program predicts a flower type based on a convolutional 
NN trained with the 102 Category Flower dataset 

'''

parser = argparse.ArgumentParser(
    description = "This program processes a flower picture and \
            predicts the species of this flower"
)

# parser options
parser.add_argument("--path_image", action="store", \
type=str, dest='path_image', default='./',\
help="Choose test image path")

parser.add_argument("--checkpoint_root", action="store", \
type=str, dest='checkpoint_root', default='./',\
help="Choose model checkpoint root path")

parser.add_argument("--top_k", action="store", \
type=int, dest='top_k', default='5',\
help="Define number of top classes printed with result")

parser.add_argument("--category_names", action="store", \
type=str, dest='category_names', default='cat_to_name.json',\
help="Choose file with dictionary of categories and names")

parser.add_argument("--gpu", action="store", \
type=bool, dest='device', default=False,\
help="Choose if GPU is used")

args = parser.parse_args()

path_image = args.path_image
checkpoint_root = args.checkpoint_root
top_k = args.top_k
category_names = args.category_names

if args.device == True:
    device = 'cuda' 
else:
    device = 'cpu'



class prediction(object):

    def __init__(self):

        with open(category_names) as json_file:
            self.cat_to_name = json.load(json_file)

    def loadModel(self, checkpoint_file, device):
   
        checkpoint_path = checkpoint_root + checkpoint_file

        checkpoint = torch.load(checkpoint_path, \
                map_location=device)

        if checkpoint['conv_model'] == 'vgg16':
            model = models.vgg16(pretrained=True)
        else:
            model = models.resnet50(pretrained=True)
    
        for param in model.parameters():
            param.requires_grad = False
        
        n_inputs = checkpoint['input_size']
        n_hidden = checkpoint['hidden_size']
        n_output = checkpoint['output_size']
    
        clf = nn.Sequential(OrderedDict([\
                ('fc1', nn.Linear(n_inputs, n_hidden[0])),\
                ('relu1', nn.ReLU()),\
                ('dropout1', nn.Dropout(p=0.2)),\
                ('fc2', nn.Linear(n_hidden[0], n_hidden[1])),\
                ('relu2', nn.ReLU()),\
                ('dropout2', nn.Dropout(p=0.2)),\
                ('fc3', nn.Linear(n_hidden[1], n_output)),\
                ('log_softmax', nn.LogSoftmax(dim=1))]))
    
        torch.nn.init.kaiming_normal_(clf.fc1.weight, \
                mode='fan_in')
        torch.nn.init.kaiming_normal_(clf.fc2.weight, \
                mode='fan_in')
        torch.nn.init.kaiming_normal_(clf.fc3.weight, \
                mode='fan_in')
    
        if checkpoint['conv_model'] == 'vgg16':
            model.classifier = clf
        else: # defaults to resnet50
            model.fc = clf
    
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
    
        return model

    def process_image(self, image):

        ''' 
            Scales, crops, and normalizes a PIL image 
            for use with a PyTorch model, 
            returns an Numpy array
        '''
    
        img = Image.open(image)
        scale = transforms.Resize(255)
        crop = transforms.CenterCrop(224)
        toTensor = transforms.ToTensor()
        return toTensor(crop(scale(img))).numpy()

    def predict(self, image, model, top_k=5):

        ''' 
            Predict the class (or classes) of an image using a 
            trained deep convolutional model.
    
        '''
    
        img = Image.open(image)
        scale = transforms.Resize(255)
        crop = transforms.CenterCrop(224)
        toTensor = transforms.ToTensor()
        norm = transforms.Normalize([0.485,0.456,0.406],\
                [0.229, 0.224, 0.225])
        img_tensor = norm(toTensor(crop(scale(img)))).\
                unsqueeze(dim=0)
        with torch.no_grad():
            output = model.forward(img_tensor)
            prob = torch.exp(output)
            top_prob, top_class = prob.topk(top_k, dim=1)
    
        top_prob = top_prob.detach().numpy().tolist()[0]
        top_class = top_class.detach().numpy().tolist()[0]
    
        return top_prob, top_class

    def plot_prediction(self, image, model, top_k = 5):
        top_p, top_class = self.predict(image, model, top_k)
        img = self.process_image(image)
        # convert index to classes
        index_to_class = {j:i for i,j in \
                model.class_to_idx.items()}
        # convert classes to labels
        topk = [self.cat_to_name[index_to_class[i]] \
                for i in top_class]
    
        # plot figure
        fig, axes = plt.subplots(figsize=(12,4), ncols = 2)
        axes[0].imshow(img.transpose())
        axes[1].barh(topk, top_p)
        plt.savefig('prediction.png')

if __name__=='__main__':

    # instantiate class predict
    pred = prediction() 

    # load pretrained model
    model = pred.loadModel('checkpoint_final.pth', device)

    img = pred.process_image('test_image.jpg')

    top_prob, top_class = pred.predict('test_image.jpg', \
            model, top_k)
    for i,j in zip(top_prob, top_class):
        print(f'class {j:} - prob {i:}')


    pred.plot_prediction('test_image.jpg', model)

