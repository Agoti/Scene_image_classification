
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from Utils import Utils
from Config import DatasetConfig, ModelConfig


class Visualize:

    def __init__(self, 
                checkpoint_path,
                checkpoint_epoch=None, 
                max_data_num=10,
                device='cpu'):
        '''
        Initialize the tester
        Args:
            checkpoint_path: The path of the checkpoint
            checkpoint_epoch: Load the model from a specific epoch milestone
            max_data_num: The maximum number of the data to visualize
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
        self.dataset_config.max_data_num = max_data_num
        self.max_data_num = max_data_num

        # Build model
        self.model = Utils.build_model(self.model_config)
        # If checkpoint_epoch is specified, load the model from a specific epoch milestone
        if checkpoint_epoch:
            weight_path = os.path.join(checkpoint_path, 'checkpoint', f'{self.model_config.model_name}_epoch_{checkpoint_epoch}.pth')
        else:
            weight_path = os.path.join(checkpoint_path, f'{self.model_config.model_name}.pth')

        # Load the model
        self.model.load_model(weight_path, map_location=device)
        self.device = device
        self.model.to(device)
        
        # Build dataset
        self.test_dataset = Utils.build_test_dataset(self.dataset_config)
        self.dataset_config.transform_name = 'default'
        self.visual_dataset = Utils.build_test_dataset(self.dataset_config)

    def grad_cam(self, image, original_image):
        '''
        Grad-CAM
        Args:
            image: The image input
            original_image: The original image without normalization
        '''
        
        # Set the model to evaluation mode
        self.model.eval()

        # Get the image
        input_tensor = image.unsqueeze(0).to(self.device)

        # Register hooks
        target_layer = self.model.features[-1].conv
        activation = None
        def hook_fn(module, input, output):
            nonlocal activation
            activation = output
        hook = target_layer.register_forward_hook(hook_fn)

        # Forward the image
        output = self.model(input_tensor)
        
        # Get the predicted label
        pred_label = output.argmax(1).item()

        # Get the gradient
        self.model.zero_grad()
        output[0, pred_label].backward()

        # Get the gradient of the target layer
        gradients = self.model.features[-1].conv.weight.grad
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # Apply the gradients
        for i in range(activation.shape[1]):
            activation[:, i, :, :] *= pooled_gradients[i]

        # Get the heatmap
        heatmap = F.relu(torch.sum(activation, dim=1)).squeeze().detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)

        # Overlay the heatmap
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        original_image = original_image.permute(1, 2, 0).numpy()
        alpha = 0.4
        overlay = cv2.addWeighted(original_image, 1, heatmap, alpha, 0)

        return overlay, pred_label
        
    
    def visualize(self):
        '''
        Visualize the Grad-CAM
        '''
        for i, (image, label) in enumerate(self.test_dataset):
            original_image, _ = self.visual_dataset[i]
            if i >= self.max_data_num:
                break
            overlay, pred_label = self.grad_cam(image, original_image)
            cv2.imshow(str(pred_label) + ' ' + str(label), overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    # Test the Visualize class
    vis = Visualize('checkpoints/AlexNet_0527_aug',
                    max_data_num=20)
    vis.visualize()
