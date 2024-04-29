
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvPoolBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 pool_layer='max_pool',
                 pool_size=2, 
                 norm_layer='batch_norm', 
                 activation='relu'):
        super(ConvPoolBlock, self).__init__()

        # Convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('Invalid activation function')

        # Pooling layer
        if pool_layer == 'max_pool':
            self.pool = nn.MaxPool2d(pool_size)
        elif pool_layer == 'avg_pool':
            self.pool = nn.AvgPool2d(pool_size)
        else:
            self.pool = None

        # Normalization layer
        if norm_layer == 'batch_norm':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_layer.startswith('local_response_norm'):
            size = int(norm_layer.split('_')[-1])
            self.norm = nn.LocalResponseNorm(size)
        else:
            self.norm = None
        
    def forward(self, x):

        # Convolutional layer
        x = self.conv(x)

        # Activation function
        x = self.activation(x)
        
        # Normalization layer
        if self.norm:
            x = self.norm(x)

        # Pooling layer
        if self.pool:
            x = self.pool(x)

        return x

class LinearBlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 norm_layer='batch_norm',
                 activation='relu'):
        super(LinearBlock, self).__init__()

        # Fully connected layer
        self.fc = nn.Linear(in_features, out_features)

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('Invalid activation function')
        
        # Normalization layer
        if norm_layer == 'batch_norm':
            self.norm = nn.BatchNorm1d(out_features)
        elif norm_layer == 'dropout':
            self.norm = nn.Dropout()
        else:
            self.norm = None
        
    def forward(self, x):

        # Fully connected layer
        x = self.fc(x)

        # Activation function
        x = self.activation(x)
        
        # Normalization layer
        if self.norm:
            x = self.norm(x)

        return x

class AlexNet(nn.Module):
    def __init__(self,
                num_classes=6):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            ConvPoolBlock(3, 64, kernel_size=11, stride=4, padding=2, pool_layer='max_pool', pool_size=3, norm_layer='local_response_norm', activation='relu'),
            ConvPoolBlock(64, 192, kernel_size=5, stride=1, padding=2, pool_layer='max_pool', pool_size=3, norm_layer='local_response_norm', activation='relu'),
            ConvPoolBlock(192, 384, kernel_size=3, stride=1, padding=1, pool_layer=None, norm_layer='batch_norm', activation='relu'),
            ConvPoolBlock(384, 256, kernel_size=3, stride=1, padding=1, pool_layer=None, norm_layer='batch_norm', activation='relu'),
            ConvPoolBlock(256, 256, kernel_size=3, stride=1, padding=1, pool_layer='max_pool', pool_size=3, norm_layer='batch_norm', activation='relu')
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            LinearBlock(256 * 6 * 6, 4096, norm_layer='batch_norm', activation='relu'),
            LinearBlock(4096, 1024, norm_layer='batch_norm', activation='relu'),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    @torch.no_grad()
    def inference(self, x):
        x = self.forward(x)
        return F.softmax(x, dim=1)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path, map_location=None):
        self.load_state_dict(torch.load(path), map_location=map_location, strict=False)

if __name__ == '__main__':
    model = AlexNet()
    print(model)
