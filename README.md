# Developing an AI application - Image Classifier

Project 2 for Udacity's Data Scientist Nanodegree

## Overview

In this project we trained an image classifier to 
recognize species of flowers from the 102 Category Flower
Dataset ([dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

The objectives of this project were:

1. Prepare a real image dataset for processing
2. Train a Convolutional Neural Network (CNN) model and 
export the network parameters
3. Develop a standalone application that can make a
top-k prediction of flower species from pictures using 
the pre-trained network

While this notebook contains both training and testing 
procedures, standalone python codes were written to 
achieve the same result. 

## Methods

We used PyTorch as the main library to build the model.
Two pre-trained CNNs, downloaded from torchvision 
models, were used: VGG-16 and ResNet-50.  

The CNNs were appended with a two-hidden-layer 
fully-connected network for classification.

Parameters for the classification network:
- He initialization
- Adam optimizer (learn rate 0.003)
- Dropout regularization (probability 0.2)
- ReLU activation
- Negative Log-Likelihood loss


## Conclusions

- VGG-16 performed better than Resnet-50, despite being
a simpler model 
for the accuracy 
- The choice of pre-trained model CNN was the most important
factor for prediction accuracy
- The test images that had not been correctly predicted 
still had the correct species in the top 5 list 
