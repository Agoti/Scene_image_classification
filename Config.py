# Config.py
# Description: This file is used to define the configuration class. Configurations are used to store the hyperparameters and settings of the model.
# Author: Mingxiao Liu

import os
import json

class Config:
    '''
    Configuration class
    Base class for different configurations
    Methods:
        __init__: Initialize the configuration
        save: Save the configuration to the file
        load: Load the configuration from the file
    '''
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value) 

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f)
    
    def load(self, path):
        with open(path, 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(self, key, value)

class DatasetConfig(Config):
    '''
    Dataset configuration class
    Attributes:
        data_dir: The directory of the data
        transform_name: The name of the transform
        max_data_num: The maximum number of the data
    '''
    
    def __init__(self,
                data_dir='./data',  
                transform_name='default', 
                max_data_num=float('inf'),
                **kwargs):
        for key, value in locals().items():
            if key not in ['self', 'kwargs'] and not key.startswith('_'):
                kwargs[key] = value
        super().__init__(**kwargs)


class ModelConfig(Config):
    '''
    Model configuration class
    Attributes:
        model_name: The name of the model
    '''

    def __init__(self,
                model_name='alexnet',
                **kwargs):
        for key, value in locals().items():
            if key not in ['self', 'kwargs'] and not key.startswith('_'):
                kwargs[key] = value
        super().__init__(**kwargs)


class TrainConfig(Config):
    '''
    Training configuration class
    Attributes:
        optimizer: Name of the optimizer
        scheduler: Name of the scheduler
        criterion: The loss function
        device: Cuda or cpu
        num_epochs: The number of epochs for training
        batch_size: Batch size for training
        learning_rate: The learning rate for updating the weights
        weight_decay: The weight decay for regularization
        step_size: The step size for the lr scheduler
        gamma: Learning rate decay factor
        checkpoint_interval: The epoch interval for saving the checkpoint
        checkpoint_dir: The directory of the checkpoint
        seed: Starting seed for random number generator
    '''
    
    def __init__(self,
                optimizer='adam',
                scheduler='step',
                criterion='cross_entropy',
                device='cuda',
                num_epochs=100,
                batch_size=32,
                learning_rate=0.001,
                weight_decay=0.0,
                step_size=30,
                gamma=0.1,
                checkpoint_interval=10,
                checkpoint_dir='./checkpoints',
                seed=0,
                **kwargs):
        for key, value in locals().items():
            if key not in ['self', 'kwargs'] and not key.startswith('_'):
                kwargs[key] = value
        super().__init__(**kwargs)


# Generate the basic configuration json files
if __name__ == '__main__':
    dataset_config = DatasetConfig()
    dataset_config.save('./config/dataset_config.json')
    dataset_config.load('./config/dataset_config.json')
    print(dataset_config.__dict__)

    model_config = ModelConfig()
    model_config.save('./config/model_config.json')
    model_config.load('./config/model_config.json')
    print(model_config.__dict__)

    train_config = TrainConfig()
    train_config.save('./config/train_config.json')
    train_config.load('./config/train_config.json')
    print(train_config.__dict__)
