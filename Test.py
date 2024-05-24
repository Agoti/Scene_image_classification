import torch
import numpy
import os
import random
import argparse
import tqdm
from Model.AlexNet import AlexNet
from Dataset import SceneDataset
from Config import DatasetConfig, ModelConfig, TrainConfig
from Metrics import Metrics

class Test:
    
    def __init__(self, 
             checkpoint_path,
             result_path=None,
             checkpoint_epoch=None, 
             batch_size=32, 
             device='cuda'):
        
        # Set seed
        self.set_seed(0)
        
        # Load configs
        config_path = os.path.join(checkpoint_path, 'config')
        self.model_config = ModelConfig()
        self.model_config.load(os.path.join(config_path, 'model_config.json'))
        self.dataset_config = DatasetConfig()
        self.dataset_config.load(os.path.join(config_path, 'dataset_config.json'))

        # Build model
        self.model = self.build_model(self.model_config)
        if checkpoint_epoch:
            weight_path = os.path.join(checkpoint_path, 'checkpoint', f'{self.model_config.model_name}_epoch_{checkpoint_epoch}.pth')
        else:
            weight_path = os.path.join(checkpoint_path, f'{self.model_config.model_name}.pth')

        self.model.load_model(weight_path)
        self.model.to(device)

        # Build dataset
        self.test_dataset = self.build_dataset(self.dataset_config)

        # Build metrics
        self.result_path = result_path
        self.metrics = Metrics(['all'])

        # Build dataloader
        self.batch_size = batch_size
        self.device = device
        self.test_dataloader = self.build_dataloader(self.test_dataset, self.batch_size, shuffle=True)
    
    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        numpy.random.seed(seed)
        random.seed(seed)
    
    @staticmethod
    def build_model(model_config):
        if model_config.model_name == 'alexnet':
            model = AlexNet()
        else:
            raise ValueError('Invalid model name')
        
        return model
    
    @staticmethod
    def build_dataset(dataset_config):
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
    
    def test(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        for images, labels in tqdm.tqdm(self.test_dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                preds = self.model.inference(images)
            all_preds.append(preds)
            all_labels.append(labels)
        
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        return all_preds, all_labels
    
    def evaluate(self):
        all_preds, all_labels = self.test()
        y_pred = torch.argmax(all_preds, dim=1).cpu().numpy()
        y_true = all_labels.cpu().numpy()
        metrics = self.metrics.compute(y_true, y_pred)
        self.metrics.save(os.path.join(self.result_path, 'metrics_test.json'))
        return metrics['overall']['accuracy']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--result_path', type=str, default=None)
    parser.add_argument('--checkpoint_epoch', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    tester = Test(args.checkpoint_path,
                  args.result_path,
                  args.checkpoint_epoch,
                  args.batch_size,
                  args.device)
    accuracy = tester.evaluate()
    print(f'Accuracy: {accuracy:.4f}')
        
