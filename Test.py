# Test.py
# Description: This file is used to test the model.
# Author: Mingxiao Liu

import torch
import os
import argparse
import tqdm
from Utils import Utils
from Config import DatasetConfig, ModelConfig
from Metrics import Metrics

class Test:
    '''
    Test class
    Methods:
        predict: Predict the results
        evaluate: Evaluate the model
    '''
    
    def __init__(self, 
             checkpoint_path,
             result_path=None,
             checkpoint_epoch=None, 
             batch_size=32, 
             device='cuda'):
        '''
        Initialize the tester
        Args:
            checkpoint_path: The path of the checkpoint
            result_path: The path to save the results
            checkpoint_epoch: Load the model from a specific epoch milestone
            batch_size: The batch size
            device: The device to use
        '''
        
        # Set seed
        Utils.set_seed(0)
        
        # Load configs from the checkpoint
        config_path = os.path.join(checkpoint_path, 'config')
        self.model_config = ModelConfig()
        self.model_config.load(os.path.join(config_path, 'model_config.json'))
        self.dataset_config = DatasetConfig()
        self.dataset_config.load(os.path.join(config_path, 'dataset_config.json'))

        # Build model
        self.model = Utils.build_model(self.model_config)
        # If checkpoint_epoch is specified, load the model from a specific epoch milestone
        if checkpoint_epoch:
            weight_path = os.path.join(checkpoint_path, 'checkpoint', f'{self.model_config.model_name}_epoch_{checkpoint_epoch}.pth')
        else:
            weight_path = os.path.join(checkpoint_path, f'{self.model_config.model_name}.pth')

        # Load the model
        self.model.load_model(weight_path)
        self.model.to(device)
        # if torch.cuda.device_count() > 1:
        #     print(f'Using {torch.cuda.device_count()} GPUs')
        #     self.model = torch.nn.DataParallel(self.model, device_ids=list(range(torch.cuda.device_count())))

        # Build dataset
        self.test_dataset = Utils.build_test_dataset(self.dataset_config)

        # Build metrics
        self.result_path = result_path
        self.metrics = Metrics(['all'])

        # Build dataloader
        self.batch_size = batch_size
        self.device = device
        self.test_dataloader = Utils.build_dataloader(self.test_dataset, self.batch_size, shuffle=True)
    
    
    def predict(self):
        '''
        Predict the results
        '''

        print('Predicting...')
        # Set the model to evaluation mode
        self.model.eval()

        # Forward pass
        all_preds = []
        all_labels = []
        for images, labels in tqdm.tqdm(self.test_dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                preds = self.model(images)
            all_preds.append(preds)
            all_labels.append(labels)

        # Concatenate the results        
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return all_preds, all_labels
    
    def evaluate(self):
        '''
        Evaluate the model
        '''

        print('Evaluating...')
        # Predict the results
        all_preds, all_labels = self.predict()
        y_pred = torch.argmax(all_preds, dim=1).cpu().numpy()
        y_true = all_labels.cpu().numpy()

        # Compute the metrics
        metrics = self.metrics.compute(y_true, y_pred)
        self.metrics.save(os.path.join(self.result_path, 'metrics_test.json'))

        return metrics['overall']['accuracy']


if __name__ == '__main__':

    # Parse the arguments for testing. 
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, default=None)
    parser.add_argument('--checkpoint_epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Build the tester and evaluate the model
    tester = Test(args.checkpoint_path,
                  args.result_path,
                  args.checkpoint_epoch,
                  args.batch_size,
                  args.device)
    accuracy = tester.evaluate()
    print(f'Accuracy: {accuracy:.4f}')
        
