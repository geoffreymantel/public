import torch
import torch.nn as nn
from collections import OrderedDict

# Input: (batch_size, 1, 64, 64)
# Output: (batch_size, latent_dim)
class EncoderConv2D(nn.Module):
    def __init__(self, latent_dim, affine=True):
        super(EncoderConv2D, self).__init__()
        layers = OrderedDict([
            ('layer1',   nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, padding='same')),
            ('lrelu1',   nn.ReLU()),
            ('norm1',    nn.BatchNorm2d(num_features=10)),
            ('maxpool1', nn.MaxPool2d(kernel_size=2)),
            ('layer2',   nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding='same')),
            ('lrelu2',   nn.ReLU()),
            ('norm2',    nn.BatchNorm2d(num_features=20)),
            ('maxpool2', nn.MaxPool2d(kernel_size=2)),
            ('layer3',   nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, padding='same')),
            ('lrelu3',   nn.ReLU()),
            ('norm3',    nn.BatchNorm2d(num_features=20)),
            ('maxpool3', nn.MaxPool2d(kernel_size=2)),
            ('flatten',  nn.Flatten(start_dim=1)),
            ('linear',   nn.Linear(in_features=1280, out_features=latent_dim)),
            ('norm',     nn.BatchNorm1d(num_features=latent_dim, affine=affine))
        ])

        self.model = nn.Sequential(layers)
        
    def forward(self, x):
        x = self.model(x)
        return x
