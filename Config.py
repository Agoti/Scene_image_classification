import os
import json

class Config:
    
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
    
    def __init__(self,
                data_dir='./data',  
                split='train',
                transform_name='default', 
                max_data_num=float('inf'),
                **kwargs):
        for key, value in locals().items():
            if key not in ['self', 'kwargs'] and not key.startswith('_'):
                kwargs[key] = value
        super().__init__(**kwargs)

class ModelConfig(Config):

    def __init__(self,
                name='alexnet',
                load_checkpoint_path=None,
                save_checkpoint_path=None,
                **kwargs):
        for key, value in locals().items():
            if key not in ['self', 'kwargs'] and not key.startswith('_'):
                kwargs[key] = value
        super().__init__(**kwargs)


class TrainConfig(Config):
    
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
                **kwargs):
        for key, value in locals().items():
            if key not in ['self', 'kwargs'] and not key.startswith('_'):
                kwargs[key] = value
        super().__init__(**kwargs)

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
