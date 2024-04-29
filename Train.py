import os
import torch
import argparse
from Model import AlexNet
from Dataset import SceneDataset
from Config import DatasetConfig, ModelConfig, TrainConfig
from sklearn.metrics import accuracy_score, average_precision_score

class Trainer:

    def __init__(self,
                 dataset_config,
                 model_config,
                 train_config):
        
        # Build dataset and dataloader
        print('Building dataset and dataloader...')
        self.train_dataset, self.val_dataset = self.build_dataset(dataset_config)
        print('Length of train dataset:', len(self.train_dataset))
        print('Length of val dataset:', len(self.val_dataset))

        # Build dataloader
        self.train_dataloader = self.build_dataloader(self.train_dataset, train_config.batch_size)
        self.val_dataloader = self.build_dataloader(self.val_dataset, train_config.batch_size, shuffle=False)
        print('Finish building dataloader...')

        # Build model
        print('Building model...')
        self.model = self.build_model(model_config)
        print('Finish building model...')

        # Build optimizer
        self.optimizer, self.scheduler = self.build_optimizer(train_config, self.model)

        # Build criterion
        self.criterion = self.build_criterion(train_config)
        
        self.device = torch.device(train_config.device)
        self.model.to(self.device)
        self.num_epochs = train_config.num_epochs
    
    @staticmethod
    def build_model(model_config):

        if model_config.name == 'alexnet':
            model = AlexNet()
        else:
            raise ValueError('Invalid model name')
        
        if model_config.checkpoint_path:
            print('Loading model checkpoint from:', model_config.checkpoint_path)
            model.load_model(model_config.checkpoint_path)
        
        return model
    
    @staticmethod
    def build_dataset(dataset_config):
        image_dir = os.path.join(dataset_config.data_dir, 'imgs')
        train_annotation_path = os.path.join(dataset_config.data_dir, 'train_data.csv')
        val_annotation_path = os.path.join(dataset_config.data_dir, 'val_data.csv')
        train_dataset = SceneDataset(train_annotation_path,
                                     image_dir,
                                     split='train',
                                     transform=dataset_config.transform,
                                     max_data_num=dataset_config.max_data_num)
        
        val_dataset = SceneDataset(val_annotation_path,
                                   image_dir,
                                   split='val',
                                   transform=dataset_config.transform,
                                   max_data_num=dataset_config.max_data_num)
        
        return train_dataset, val_dataset
    
    @staticmethod
    def build_dataloader(dataset, batch_size, shuffle=True):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    @staticmethod
    def build_transform():
        return None
    
    @staticmethod
    def build_optimizer(train_config, model):

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

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for images, labels in self.train_dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        return total_loss / len(self.train_dataset)
    
    def train(self):

        checkpoint_interval = self.train_config.checkpoint_interval
        checkpoint_dir = self.train_config.checkpoint_dir
        checkpoint_subdir = os.path.join(checkpoint_dir, 'checkpoint')

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        if not os.path.exists(checkpoint_subdir):
            os.makedirs(checkpoint_subdir)

        self.model.train()
        for epoch in range(self.num_epochs):
            loss = self.train_one_epoch()
            acc, ap = self.validate()
            print('Epoch:', epoch, 'Loss:', loss, 'Acc:', acc, 'AP:', ap)
            if epoch % checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_subdir, f'_epoch_{epoch}.pth')
                self.model.save_model(checkpoint_path)
                print('Checkpoint saved to:', checkpoint_path)
        
        print('Training finished...')
        
    def validate(self):
        self.model.eval()
        total_correct = 0
        predicted_labels = []
        with torch.no_grad():
            for images, labels in self.val_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                predicted_labels.extend(predicted.cpu().numpy())
        
        gt_labels = self.val_dataset.labels
        acc = accuracy_score(gt_labels, predicted_labels)
        ap = average_precision_score(gt_labels, predicted_labels)
        return acc, ap

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
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
    parser.add_argument('--load_from', type=str, default=None)
    args = parser.parse_args()
    
    if args.load_from:
        dataset_config = DatasetConfig()
        model_config = ModelConfig()
        train_config = TrainConfig()
        dataset_config.load(os.path.join(args.load_from, 'dataset_config.json'))
        model_config.load(os.path.join(args.load_from, 'model_config.json'))
        train_config.load(os.path.join(args.load_from, 'train_config.json'))
    else:
        dataset_config = DatasetConfig()
        dataset_config.load(os.path.join(parser.config_dir, 'dataset_config.json'))
        model_config = ModelConfig()
        model_config.load(os.path.join(parser.config_dir, 'model_config.json'))
        train_config = TrainConfig()
        train_config.load(os.path.join(parser.config_dir, 'train_config.json'))

    for cfg in (dataset_config, model_config, train_config):
        for key, value in vars(args).items():
            setattr(cfg, key, value)

    trainer = Trainer(dataset_config, model_config, train_config)
    if args.load_from:
        model_path = os.path.join(args.load_from, model_config.model_name + '.pth')
        trainer.model.load_model(model_path)
        print('Model loaded from:', model_path)

    trainer.train(train_config.num_epochs)
    acc, ap = trainer.validate()
    print('Validation Acc:', acc, 'Validation AP:', ap)
    trainer.model.save_model(model_config.model_name + '.pth')
    print('Model saved to:', model_config.model_name + '.pth')
