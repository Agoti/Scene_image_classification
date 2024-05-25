# Model.py
# Description: This file is used to define the model interface.
# Author: Mingxiao Liu

import abc
import torch
import torch.nn as nn

class Model(abc.ABC, nn.Module):
    '''
    Model interface
    Methods:
        forward: Forward the input through the model
        inference: Inference the input through the model
        save_model: Save the model to the file
        load_model: Load the model from the file
    '''

    def __init__(self):
        super(Model, self).__init__()

    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def inference(self, x):
        pass

    @abc.abstractmethod
    def save_model(self, path):
        pass

    @abc.abstractmethod
    def load_model(self, path):
        pass   
