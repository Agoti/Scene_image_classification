
from torch.utils.data import Dataset
from torchvision import transforms
import os
import cv2
import numpy as np


class SceneDataset(Dataset):
    def __init__(self,
                 annotation_path,
                 image_dir,
                 split='train',
                 transform_name='default',
                 max_data_num=float('inf')):
        self.images = []
        self.labels = []
        self.split = split
        self.transform = self.build_transform(transform_name)
        self.max_data_num = max_data_num
        self.load_data(annotation_path, image_dir)
    
    @staticmethod
    def build_transform(transform_name):

        if transform_name == 'default':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError('Invalid transform name')

    def load_data(self, annotation_path, image_dir):
        
        # Read annotation file(csv file)
        with open(annotation_path, 'r') as f:
            lines = f.readlines()

        # Load image and label
        # Skip the first line because it is the header
        for i, line in enumerate(lines[1:]):
            if i >= self.max_data_num:
                break
            image_name, label = line.strip().split(',')
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            self.images.append(image)
            self.labels.append(int(label))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    dataset = SceneDataset('E:/scene_classification/test_data.csv', 
                            'E:/scene_classification/imgs',
                            split='test', 
                            max_data_num=10) 
    print(len(dataset))
    image, label = dataset[0]
    print(image.shape, label)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
