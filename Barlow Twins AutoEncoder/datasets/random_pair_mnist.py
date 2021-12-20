import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

# A dataset based on MNIST which returns two 28x28 digits, placed on random locations within a 64x64 pixel frame
# The labels are returned as a pair.
class RandomPairMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train, device):
        super(RandomPairMNIST, self).__init__()
        
        # Save device
        self.device = device

        # Not all classes in the original MNIST dataset have the same number of samples
        # We will choose 5K samples from each class in training, and 800 samples from each class in testing,
        # so that all classes will be equally represented
        transform = transforms.Compose([transforms.ToTensor()])
        if train:
            num_samples = 5000
            dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        else:
            num_samples = 800
            dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)

        # Save number of samples
        self.num_samples = num_samples
        
        # Select (num_samples, 10, 1, 28, 28) samples
        selected_samples = torch.stack([dataset.data[(dataset.targets == label).nonzero()][:num_samples] for label in range(10)])
        selected_samples = torch.swapaxes(selected_samples, 0, 1)
        selected_samples = selected_samples / 255.
        self.selected_samples = selected_samples.to(device)

        # Total num_samples x 10 x 10 pairs
        self.data = torch.zeros(size=(self.num_samples, 10, 10, 1, 64, 64), device=self.device)
        self.labels = torch.zeros(size=(self.num_samples, 10, 10, 10), device=self.device)
        
        # Initial shuffle
        self.shuffle_digit_arrangements()

    # Digits are arranged randomly on a 64x64 background.
    # The arrangements can be reshuffled.
    def shuffle_digit_arrangements(self):

        # Zero out tensors
        self.data.zero_()
        self.labels.zero_()
        
        # Reshape
        self.data = self.data.reshape((self.num_samples, 10, 10, 1, 64, 64))
        self.labels = self.labels.reshape((self.num_samples, 10, 10, 10))
        
        # Populate the classes, randomly arranged on a 64x64 background
        # Perhaps this can be done without looping, but this works and is straightforward
        random_xy = torch.randint(64-28, size=(self.num_samples, 10, 10, 4), device=self.device)
        
        for i in range(self.num_samples):
            for j in range(10):
                for k in range(10):
                    x1 = random_xy[i, j, k, 0]
                    y1 = random_xy[i, j, k, 1]
                    x2 = random_xy[i, j, k, 2]
                    y2 = random_xy[i, j, k, 3]
                    self.data[i, j, k, 0, x1:x1+28, y1:y1+28] = self.selected_samples[i, j, 0, :, :]
                    self.data[i, j, k, 0, x2:x2+28, y2:y2+28] = torch.max(self.data[i, j, k, 0, x2:x2+28, y2:y2+28],
                                                                          self.selected_samples[i, k, 0, :, :])

                    # Instead of index labels, we will use probability labels
                    self.labels[i, j, k, j] += 0.5
                    self.labels[i, j, k, k] += 0.5

        # Now flatten the dataset -- the dataloader will decide if it should randomize or not
        self.data = self.data.flatten(end_dim=2)
        self.labels = self.labels.flatten(end_dim=2)
                        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return self.data[index], self.labels[index]