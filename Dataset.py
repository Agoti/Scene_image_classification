# Dataset.py
# Description: This file is used to load the dataset and preprocess the data.
# Author: Mingxiao Liu

from torch.utils.data import Dataset
from torchvision import transforms
import os
import cv2
import numpy as np


class SceneDataset(Dataset):
    '''
    Scene dataset class
    Methods:
        __init__: Initialize the dataset
        build_transform: (Static method) Build the transform for the dataset
        load_data: Load the data from the annotation file
        __len__: Get the length of the dataset
        __getitem__: Get the item of the dataset
    '''

    def __init__(self,
                 annotation_path,
                 image_dir,
                 split='train',
                 transform_name='default',
                 max_data_num=float('inf')):
        '''
        Initialize the dataset
        Args:
            annotation_path: The path of the annotation file
            image_dir: The directory of the images
            split: The split of the dataset
            transform_name: The name of the transform
            max_data_num: The maximum number of the data
        '''

        super(SceneDataset, self).__init__()
        self.images = []
        self.labels = []
        self.split = split
        self.transform = self.build_transform(transform_name)
        self.max_data_num = max_data_num
        self.load_data(annotation_path, image_dir)

    
    @staticmethod
    def build_transform(transform_name):
        '''
        Build the [transform] for the dataset
        '''

        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
        # Default transform: Resize and to tensor
        if transform_name == 'default':
            pass
        elif transform_name == 'augmentation':
            # Random crop and horizontal flip and normalize and ...
            transform_list.insert(2, transforms.RandomHorizontalFlip())
            transform_list.insert(2, transforms.RandomCrop(224, padding=4))
            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]))
        else:
            raise ValueError('Invalid transform name')
        
        return transforms.Compose(transform_list)


    def load_data(self, annotation_path, image_dir):
        '''
        Load the data from the annotation file
        Args:
            annotation_path: The path of the annotation file
            image_dir: The directory of the images
        '''
        
        # Read annotation file(csv file)
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        # Load image and label
        # Skip the first line because it is the header
        for i, line in enumerate(lines[1:]):

            # If the number of data is greater than the max_data_num, break
            if i >= self.max_data_num:
                break

            # Get the image name and label
            image_name, label = line.strip().split(',')

            # Load the image
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            self.images.append(image)
            self.labels.append(int(label))
        

    def __len__(self):
        '''
        Get the length of the dataset. 
        Rewritten the __len__ method of the Dataset class
        '''
        return len(self.images)


    def __getitem__(self, idx):
        '''
        Get the idx-th item of the dataset
        Rewritten the __getitem__ method of the Dataset class
        '''

        # Get the image and label
        image = self.images[idx]
        label = self.labels[idx]

        # Transform the image
        if self.transform:
            image = self.transform(image)

        return image, label


# Test the dataset
if __name__ == '__main__':
    dataset = SceneDataset('E:/scene_classification/test_data.csv', 
                            'E:/scene_classification/imgs',
                            split='test',
                            transform_name='augmentation', 
                            max_data_num=10) 
    print(len(dataset))
    image, label = dataset[0]
    print(image.shape, label)
    for i in range(10):
        image, label = dataset[i]
        image = image.permute(1, 2, 0).numpy()
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
