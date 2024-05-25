# Utils.py
# Description: This file is used to define the utility functions.
# Author: Mingxiao Liu

import os
import random
import numpy
import torch
from Model.AlexNet import AlexNet, AlexNetNorm, AlexNetPretrained
from Dataset import SceneDataset
from Config import DatasetConfig, ModelConfig, TrainConfig

class Utils:

    '''
    Utility class
    Methods:
        set_seed: Set the seed for reproducibility
        build_model: Build the model
        build_dataset: Build the dataset
        build_dataloader: Build the dataloader
        build_transform: Build the transform
        build_optimizer: Build the optimizer
        build_criterion: Build the criterion
    '''

    @staticmethod
    def set_seed(seed):
        '''
        Set the seed for reproducibility
        '''

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        numpy.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    @staticmethod
    def build_model(model_config):
        '''
        Build the model from the configuration
        '''

        # Model dictionary
        models = {
            'alexnet': AlexNet,
            'alexnet_norm': AlexNetNorm,
            'alexnet_pretrained': AlexNetPretrained
        }

        if model_config.model_name in models:
            model = models[model_config.model_name]()
        else:
            raise ValueError('Invalid model name')
        
        return model
    
    @staticmethod
    def build_train_val_dataset(dataset_config):
        image_dir = os.path.join(dataset_config.data_dir, 'imgs')
        train_annotation_path = os.path.join(dataset_config.data_dir, 'train_data.csv')
        val_annotation_path = os.path.join(dataset_config.data_dir, 'val_data.csv')
        train_dataset = SceneDataset(train_annotation_path,
                                     image_dir,
                                     split='train',
                                     transform_name=dataset_config.transform_name,
                                     max_data_num=dataset_config.max_data_num)
        
        val_dataset = SceneDataset(val_annotation_path,
                                   image_dir,
                                   split='val',
                                   transform_name=dataset_config.transform_name,
                                   max_data_num=dataset_config.max_data_num)
        
        return train_dataset, val_dataset
    
    def build_test_dataset(dataset_config):
        image_dir = os.path.join(dataset_config.data_dir, 'imgs')
        test_annotation_path = os.path.join(dataset_config.data_dir, 'test_data.csv')
        test_dataset = SceneDataset(test_annotation_path,
                                    image_dir,
                                    split='test',
                                    transform_name=dataset_config.transform_name,
                                    max_data_num=dataset_config.max_data_num)

        return test_dataset
    
    @staticmethod
    def build_dataloader(dataset, batch_size, shuffle=True):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    @staticmethod
    def build_transform():
        return None
    
    @staticmethod
    def build_optimizer_scheduler(train_config, model):

        if train_config.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters())
        else:
            raise ValueError('Invalid optimizer name')
        
        if train_config.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        else:
            scheduler = None
        
        return optimizer, scheduler
    
    @staticmethod
    def build_criterion(train_config):
        if train_config.criterion == 'cross_entropy':
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError('Invalid criterion name')
        
        return criterion
