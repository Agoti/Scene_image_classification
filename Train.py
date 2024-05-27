# Train.py: Train the model
# Description: This file is used to train the model.
# Author: Mingxiao Liu

import os
import torch
import numpy as np
import random
import argparse
import tqdm
from Utils import Utils
from Config import DatasetConfig, ModelConfig, TrainConfig
from Metrics import Metrics

class Trainer:
    '''
    Trainer class
    Methods:
        train_one_epoch: Train the model for one epoch
        train: Train the model
        validate: Validate the model
        save: Save the model
    '''

    def __init__(self,
                 dataset_config,
                 model_config,
                 train_config):
        
        # Set random seed
        Utils.set_seed(train_config.seed)
        
        # Store configs
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.train_config = train_config
        print('Configs:')
        print('Dataset config:', dataset_config.__dict__)
        print('Model config:', model_config.__dict__)
        print('Train config:', train_config.__dict__)
        print('-' * 50)
        
        # Build dataset and dataloader
        print('Building dataset and dataloader...')
        self.train_dataset, self.val_dataset = Utils.build_train_val_dataset(dataset_config)
        print('Length of train dataset:', len(self.train_dataset))
        print('Length of val dataset:', len(self.val_dataset))
        print('-' * 50)

        # Build dataloader
        self.train_dataloader = Utils.build_dataloader(self.train_dataset, train_config.batch_size, shuffle=True, seed = train_config.seed)
        self.val_dataloader = Utils.build_dataloader(self.val_dataset, train_config.batch_size, shuffle=False, seed = train_config.seed)
        print('Finish building dataloader...')

        # Build model
        print('Building model...')
        self.model = Utils.build_model(model_config)
        print('Model:' + model_config.model_name)

        # Build optimizer and scheduler
        self.optimizer, self.scheduler = Utils.build_optimizer_scheduler(train_config, self.model)

        # Build criterion
        self.criterion = Utils.build_criterion(train_config)

        # Build metrics
        self.metrics = Metrics(['all'])
        
        # Set device
        self.device = torch.device(train_config.device)
        self.model.to(self.device)
        # multi-gpu
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     self.model = torch.nn.DataParallel(self.model)

        # Print options
        print('-' * 50)
        print('Augmentation:', 'ON' if dataset_config.transform_name == 'augmentation' else 'OFF')
        print('Learning rate:', train_config.learning_rate, '; Batch size:', train_config.batch_size, '; Scheduler:', train_config.scheduler, '; Weight decay:', train_config.weight_decay)
        print('Optimizer:', train_config.optimizer)
        print('Normalization:', 'ON' if 'norm' in model_config.model_name else 'OFF')
        print('-' * 50)


    def train_one_epoch(self):
        '''
        Train the model for one epoch
        '''

        # Set the model to train mode
        self.model.train()

        # Initialize the total loss
        total_loss = 0
        total_correct = 0

        # Iterate over the dataloader
        with tqdm.tqdm(self.train_dataloader, desc='Training') as t:

            # Iterate over the minibatches
            for i, (images, labels) in enumerate(t): 

                # Move data to device
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)
                total_correct += (torch.argmax(outputs, 1) == labels).sum().item()

                # Compute the loss
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Update the weights
                self.optimizer.step()

                # Update the total loss
                total_loss += loss.item()

                # Update the progress bar
                t.set_postfix({'loss': total_loss / (i + 1)})

        # Return the average loss and the learning rate
        return total_loss / len(self.train_dataloader), self.optimizer.param_groups[0]['lr'], total_correct / len(self.train_dataset)

    
    def train(self):
        '''
        Train the model
        '''

        # Set up the checkpoint directory
        checkpoint_interval = self.train_config.checkpoint_interval
        checkpoint_dir = self.train_config.checkpoint_dir
        checkpoint_subdir = os.path.join(checkpoint_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not os.path.exists(checkpoint_subdir):
            os.makedirs(checkpoint_subdir)

        # Train the model
        for epoch in range(1, self.train_config.num_epochs + 1):

            # Train one epoch
            loss, lr, train_acc = self.train_one_epoch()

            # Update the scheduler
            if self.scheduler:
                self.scheduler.step()

            # Validate
            acc = self.validate()
            print('Epoch:', epoch, 'Loss: %.4f' % loss, 'Val acc: %.4f' % acc, 'Train acc: %.4f' % train_acc, 'LR:', lr)

            # Save checkpoint
            if epoch % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_subdir, f'{self.model_config.model_name}_epoch_{epoch}.pth')
                self.model.save_model(checkpoint_path)
                optimizer_path = os.path.join(checkpoint_subdir, f'{self.model_config.model_name}_epoch_{epoch}_optimizer.pth')
                torch.save(self.optimizer.state_dict(), optimizer_path)
                metrics_path = os.path.join(checkpoint_subdir, f'{self.model_config.model_name}_epoch_{epoch}_metrics.json')
                self.metrics.save(metrics_path)
                print('Checkpoint saved to:', checkpoint_path)
        
        print('Training finished...')

        
    def validate(self):
        '''
        Validate the model
        '''

        # Set the model to eval mode
        self.model.eval()

        # Predict the labels
        predicted_labels = []
        with torch.no_grad():
            for images, labels in self.val_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                predicted_labels.extend(predicted.cpu().numpy())

        # Compute the metrics        
        gt_labels = self.val_dataset.labels
        # Convert predicted labels and gt labels to numpy array
        predicted_labels = np.array(predicted_labels)
        gt_labels = np.array(gt_labels)
        metrics = self.metrics.compute(gt_labels, predicted_labels)

        return metrics['overall']['accuracy']

    
    def save(self):
        '''
        Save the model after training
        '''

        # Save the model
        model_path = os.path.join(self.train_config.checkpoint_dir, self.model_config.model_name + '.pth')
        self.model.save_model(model_path)

        # Save the optimizer
        optimizer_path = os.path.join(self.train_config.checkpoint_dir, self.model_config.model_name + '_optimizer.pth')
        torch.save(self.optimizer.state_dict(), optimizer_path)

        # Save the configs
        config_path = os.path.join(self.train_config.checkpoint_dir, 'config')
        if not os.path.exists(config_path):
            os.makedirs(config_path)
        self.dataset_config.save(os.path.join(config_path, 'dataset_config.json'))
        self.model_config.save(os.path.join(config_path, 'model_config.json'))
        self.train_config.save(os.path.join(config_path, 'train_config.json'))

        # Save the metrics
        metric_path = os.path.join(self.train_config.checkpoint_dir, 'metrics.json')
        self.metrics.save(metric_path)

if __name__ == '__main__':

    # Parse the arguments. The configuration files are loaded from the config directory by default, but can be overridden by the command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--transform_name', type=str, default='default')
    parser.add_argument('--config_dir', type=str, default='./config')
    parser.add_argument('--model_name', type=str, default='alexnet')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/checkpoint')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--load_from', type=str, default=None)
    args = parser.parse_args()
    
    # If the model is loaded from a checkpoint, load the configs from the checkpoint directory. Otherwise, load the configs from the config directory.
    if args.load_from:
        dataset_config = DatasetConfig()
        model_config = ModelConfig()
        train_config = TrainConfig()
        dataset_config.load(os.path.join(args.load_from, 'dataset_config.json'))
        model_config.load(os.path.join(args.load_from, 'model_config.json'))
        train_config.load(os.path.join(args.load_from, 'train_config.json'))
    else:
        dataset_config = DatasetConfig()
        model_config = ModelConfig()
        train_config = TrainConfig()
        model_config.load(os.path.join(args.config_dir, 'model_config.json'))
        dataset_config.load(os.path.join(args.config_dir, 'dataset_config.json'))
        train_config.load(os.path.join(args.config_dir, 'train_config.json'))

    # Override the configs with the command line arguments
    for cfg in (dataset_config, model_config, train_config):
        for key, value in vars(args).items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    # Create the trainer
    trainer = Trainer(dataset_config, model_config, train_config)

    # Load the model from the checkpoint if specified
    if args.load_from:
        model_path = os.path.join(args.load_from, model_config.model_name + '.pth')
        trainer.model.load_model(model_path)
        optimizer_path = os.path.join(args.load_from, model_config.model_name + '_optimizer.pth')
        trainer.optimizer.load_state_dict(torch.load(optimizer_path))
        print('Model loaded from:', model_path)

    # Train the model
    trainer.train()
    acc = trainer.validate()
    print('Validation Acc:', acc)
    trainer.save()
