# AlexNet.py
# Description: This file is used to define the AlexNet model.
# Author: Mingxiao Liu

import abc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet
from Model import Model


class ConvPoolBlock(nn.Module):
    '''
    Convolutional and pooling block
    Methods:
        __init__: Initialize the ConvPoolBlock
        forward: Forward the input through the block
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 pool_layer='max_pool',
                 pool_size=2, 
                 norm_layer='batch_norm', 
                 activation='relu'):
        '''
        Initialize the ConvPoolBlock
        Args:
            in_channels: The number of input channels
            out_channels: The number of output channels
            kernel_size: The size of the kernel
            stride: The stride of the convolution
            padding: The padding of the convolution
            pool_layer: The type of the pooling layer
            pool_size: The size of the pooling layer
            norm_layer: The type of the normalization layer
            activation: The type of the activation function
        '''

        # Call the constructor of the parent class
        super(ConvPoolBlock, self).__init__()

        # Convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('Invalid activation function')

        # Pooling layer
        if pool_layer == 'max_pool':
            self.pool = nn.MaxPool2d(pool_size)
        elif pool_layer == 'avg_pool':
            self.pool = nn.AvgPool2d(pool_size)
        else:
            self.pool = None

        # Normalization layer
        if norm_layer == 'batch_norm':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_layer and norm_layer.startswith('local_response_norm'):
            size = norm_layer.split('_')[-1]
            size = int(size) if size.isdigit() else 5
            self.norm = nn.LocalResponseNorm(size)
        elif norm_layer and norm_layer.startswith('dropout'):
            prob = norm_layer.split('_')[-1]
            prob = float(prob) if prob.isdigit() else 0.5
            self.norm = nn.Dropout(prob)
        else:
            self.norm = None

        
    def forward(self, x):

        # Convolutional layer
        x = self.conv(x)

        # Activation function
        x = self.activation(x)
        
        # Normalization layer
        if self.norm:
            x = self.norm(x)

        # Pooling layer
        if self.pool:
            x = self.pool(x)

        return x


class LinearBlock(nn.Module):
    '''
    Fully connected block
    Methods:
        __init__: Initialize the LinearBlock
        forward: Forward the input through the block
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 norm_layer='batch_norm',
                 activation='relu'):
        '''
        Initialize the LinearBlock
        Args:
            in_features: The number of input features
            out_features: The number of output features
            norm_layer: The type of the normalization layer
            activation: The type of the activation function
        '''

        # Call the constructor of the parent class
        super(LinearBlock, self).__init__()

        # Fully connected layer
        self.fc = nn.Linear(in_features, out_features)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('Invalid activation function')
        
        # Normalization layer
        if norm_layer == 'batch_norm':
            self.norm = nn.BatchNorm1d(out_features)
        elif norm_layer and norm_layer.startswith('dropout'):
            prob = norm_layer.split('_')[-1]
            prob = float(prob) if prob.isdigit() else 0.5
            self.norm = nn.Dropout(prob)
        else:
            self.norm = None
        
    def forward(self, x):

        # Fully connected layer
        x = self.fc(x)

        # Activation function
        x = self.activation(x)
        
        # Normalization layer
        if self.norm:
            x = self.norm(x)

        return x


class AlexNet(Model):
    '''
    AlexNet model
    Without normalization layers
    Methods:
        __init__: Initialize the AlexNet
        forward: Forward the input through the model
        inference: Inference the input through the model
        save_model: Save the model to the file
        load_model: Load the model from the file
    '''
    def __init__(self,
                num_classes=6):
        super(AlexNet, self).__init__()

        # Convolutional and pooling layers
        self.features = nn.Sequential(
            ConvPoolBlock(3, 64, kernel_size=11, stride=4, padding=2, pool_layer='max_pool', pool_size=3, norm_layer=None, activation='relu'),
            ConvPoolBlock(64, 192, kernel_size=5, stride=1, padding=2, pool_layer='max_pool', pool_size=3, norm_layer=None, activation='relu'),
            ConvPoolBlock(192, 384, kernel_size=3, stride=1, padding=1, pool_layer=None, norm_layer=None, activation='relu'),
            ConvPoolBlock(384, 256, kernel_size=3, stride=1, padding=1, pool_layer=None, norm_layer=None, activation='relu'),
            ConvPoolBlock(256, 256, kernel_size=3, stride=1, padding=1, pool_layer='max_pool', pool_size=3, norm_layer=None, activation='relu')
        )

        # Dynamic pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Fully connected layers
        self.classifier = nn.Sequential(
            LinearBlock(256 * 6 * 6, 4096, norm_layer='dropout', activation='relu'),
            LinearBlock(4096, 1024, norm_layer='dropout', activation='relu'),
            nn.Linear(1024, num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    
    @torch.no_grad()
    def inference(self, x):
        x = self.forward(x)
        return F.softmax(x, dim=1)
    

    def save_model(self, path):
        print("AlexNet: Saving model to", path)
        torch.save(self.state_dict(), path)
    

    def load_model(self, path):
        print("AlexNet: Loading model from", path)
        self.load_state_dict(torch.load(path), strict=False)


class AlexNetNorm(AlexNet):
    '''
    AlexNet model with normalization layers
    '''

    def __init__(self,
                num_classes=6):
        super(AlexNetNorm, self).__init__()

        # The only difference is the normalization layers
        self.features = nn.Sequential(
            ConvPoolBlock(3, 64, kernel_size=11, stride=4, padding=2, pool_layer='max_pool', pool_size=3, norm_layer='local_response_norm_5', activation='relu'),
            ConvPoolBlock(64, 192, kernel_size=5, stride=1, padding=2, pool_layer='max_pool', pool_size=3, norm_layer='local_response_norm_5', activation='relu'),
            ConvPoolBlock(192, 384, kernel_size=3, stride=1, padding=1, pool_layer=None, norm_layer=None, activation='relu'),
            ConvPoolBlock(384, 256, kernel_size=3, stride=1, padding=1, pool_layer=None, norm_layer=None, activation='relu'),
            ConvPoolBlock(256, 256, kernel_size=3, stride=1, padding=1, pool_layer='max_pool', pool_size=3, norm_layer=None, activation='relu')
        )


class AlexNetPretrained(Model):
    '''
    AlexNet model with pretrained weights
    '''

    def __init__(self,
                num_classes=6):

        super(AlexNetPretrained, self).__init__()
        self.model = alexnet(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    @torch.no_grad()
    def inference(self, x):
        x = self.forward(x)
        return F.softmax(x, dim=1)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path), strict=False)


if __name__ == '__main__':
    model = AlexNet()
    print(model)
