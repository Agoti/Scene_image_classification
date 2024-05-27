
import os
import cv2
import numpy as np
import torch
from torchcam.cams import GradCAM
from torchcam.utils import overlay_mask
from Utils import Utils
from Config import DatasetConfig, ModelConfig


class Visualize:

    def __init__(self, 
                checkpoint_path,
                checkpoint_epoch=None, 
                device='cpu'):
        '''
        Initialize the tester
        Args:
            checkpoint_path: The path of the checkpoint
            checkpoint_epoch: Load the model from a specific epoch milestone
            device: The device to use
        '''
        
        # Set seed
        Utils.set_seed(0)
        
        # Load configs from the checkpoint
        config_path = os.path.join(checkpoint_path, 'config')
        self.model_config = ModelConfig()
        self.model_config.load(os.path.join(config_path, 'model_config.json'))
        self.dataset_config = DatasetConfig()
        self.dataset_config.max_data_num = 10
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

        self.test_dataloader = Utils.build_dataloader(self.test_dataset, self.batch_size, shuffle=True)
    
    def grad_cam(self, image):
        '''
        Grad-CAM
        Args:
            image: The image to visualize
        '''
        
        # Set the model to evaluation mode
        self.model.eval()

        # Get the image
        image = image.to(self.device)
        image = image.unsqueeze(0)

        # Get the Grad-CAM
        cam = GradCAM(model=self.model)
        activation_map = cam(image)
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        activation_map = activation_map.squeeze(0).cpu().numpy()
        heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), cv2.COLORMAP_JET)
        overlay = overlay_mask(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), heatmap)

        return overlay
    
    def visualize(self):
        '''
        Visualize the results
        '''
        for i, (image, label) in enumerate(self.test_dataloader):
            if i > 10:
                break
            overlay = self.grad_cam(image)
            cv2.imshow('Grad-CAM', overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    # Test the Visualize class
    vis = Visualize('checkpoint', checkpoint_epoch=10)
    vis.visualize()
