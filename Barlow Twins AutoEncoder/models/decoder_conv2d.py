import torch
import torch.nn as nn
from collections import OrderedDict

# A little helper class
class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape
        
    def forward(self, x):
        x = x.view(*self.shape)
        return x
    
# Input: (batch_size, latent_dim)
# Output: (batch_size, 1, 64, 64)
class DecoderConv2D(nn.Module):
    def __init__(self, latent_dim):
        super(DecoderConv2D, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('linear1',   nn.Linear(in_features=latent_dim, out_features=512)),
            ('lrelu1',    nn.ReLU()),
            ('norm1',     nn.BatchNorm1d(num_features=512)),
            ('linear2',   nn.Linear(in_features=512, out_features=1280)),
            ('lrelu2',    nn.ReLU()),
            ('norm2',     nn.BatchNorm1d(num_features=1280)),
            ('reshape',   Reshape(shape=(-1, 20, 8, 8))),
            ('upsample1', nn.Upsample(scale_factor=2)),
            ('layer1',    nn.ConvTranspose2d(in_channels=20, out_channels=20, kernel_size=3, padding=1)),
            ('lrelu3',    nn.ReLU()),
            ('upsample2', nn.Upsample(scale_factor=2)),
            ('layer2',    nn.ConvTranspose2d(in_channels=20, out_channels=10, kernel_size=3, padding=1)),
            ('lrelu4',    nn.ReLU()),
            ('upsample3', nn.Upsample(scale_factor=2)),
            ('layer3',    nn.ConvTranspose2d(in_channels=10, out_channels=1, kernel_size=3, padding=1)),
            ('sigmoid',   nn.Sigmoid())
        ]))
        
    def forward(self, x):
        x = self.model(x)
        return x
