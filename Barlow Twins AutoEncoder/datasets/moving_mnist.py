# Dataset encapsulates the numpy array "mnist_test_seq.npy", copied from 
# https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
# (see http://arxiv.org/abs/1502.04681)
#
# Should contain 10K sequences of length 20, showing 2 digits moving in a 64x64 frame
#
import torch
import torchvision.transforms as transforms
import numpy as np
import os

# The original "Moving MNIST" datasource class
# Returns 10K video sequences of 20 64x64 frames each, with a 90/10 train/test split
class MovingMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train, device, split=1000):
        super(MovingMNIST, self).__init__()
        mnist_test_seq = np.load(os.path.join(root, 'mnist_test_seq.npy')).swapaxes(0, 1).astype(np.float32)
        if train:
            self.data = torch.from_numpy(mnist_test_seq[:-split])
        else:
            self.data = torch.from_numpy(mnist_test_seq[-split:])

        self.data /= 255.0
        self.data = self.data.to(device)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return self.data[index]

# Modification of the "Moving MNIST" dataset to return a pair of random frames from the same video
# For each frame in the video, pair it with the adjacent frame of the same video.
# The last frame will be paired with the first, but we hope the discontinuity won't matter much.
class MovingMNISTPairs(torch.utils.data.Dataset):
    def __init__(self, root, train, device, split=1000):
        super(MovingMNISTPairs, self).__init__()
        mnist_test_seq = np.load(os.path.join(root, 'mnist_test_seq.npy')).swapaxes(0, 1).astype(np.float32)
        if train:
            self.data = torch.from_numpy(mnist_test_seq[:-split])
        else:
            self.data = torch.from_numpy(mnist_test_seq[-split:])

        self.data /= 255.0
        self.data = self.data.reshape(-1, 1, 64, 64)
        self.data = self.data.to(device)

        # Compute indexes of adjacent pairs of frames
        self.compute_pair_index()

    # Pair adjacent frames from the same video.
    def compute_pair_index(self):
        num_videos = (int)(len(self.data) / 20)

        # Adjacent frame indexes
        frame_index = torch.tensor([(i%19)+1 for i in range(20)]).repeat(num_videos)

        # The video data has been flattened, so video 0 is the first 20 frames of the data,
        # video 1 is the next 20 frames, etc.
        video_index = (torch.tensor(range(num_videos))*20).repeat_interleave(20)

        # Adding the frame_index with the sequential video_index gives us the index for the pair
        self.pair_index = frame_index + video_index
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return self.data[index], self.data[self.pair_index[index]]


# The original "Moving MNIST" datasource class, but with transforms of the original 28x28 digit
# to random locations within a 64x64-pixel frame.
#
# Each item of data will be a list of 4 elements:
#  - a digit in a 64x64 frame
#  - the same digit in a different location of the 64x64 frame
#  - the same digit blurred
#  - the label of the digit
#
# If requested, the digits in the dataset can consist of a single given class
# This is done to construct batches in which each set of frames of the batch are in a different class

class SingleDigitRandomPairMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train, device):
        super(SingleDigitRandomPairMNIST, self).__init__()

        # Store digits in-memory according to their classes
        # Not all classes have the same number of samples
        # We will choose 5K samples in training, and 800 samples in testing, so that
        # all classes will be equally represented
        transform = transforms.Compose([transforms.ToTensor()])
        if train:
            num_samples = 5000
            self.dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        else:
            num_samples = 800
            self.dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)

        # Select (num_samples, 10, 1, 28, 28) samples
        selected_samples = torch.stack([self.dataset.data[(self.dataset.targets == label).nonzero()][:num_samples] for label in range(10)])
        selected_samples = torch.swapaxes(selected_samples, 0, 1)
        selected_samples = selected_samples / 255.
        selected_samples = selected_samples.to(device)

        # Randomly arrange 28x28 sample on a 64x64 background - twice to have 2 frames in total
        self.frame1 = torch.zeros(size=(num_samples, 10, 1, 64, 64)).to(device)
        self.frame2 = torch.zeros(size=(num_samples, 10, 1, 64, 64)).to(device)
        
        for frame in [self.frame1, self.frame2]:
            random_xy = torch.randint(64-28, size=(num_samples, 10, 2))

            # Perhaps this can be done without looping, but this works and is straightforward
            for i in range(num_samples):
                for j in range(10):
                    x = random_xy[i, j, 0]
                    y = random_xy[i, j, 1]
                    frame[i, j, 0, x:x+28, y:y+28] = selected_samples[i, j, 0, :, :]        
 
        # Finally, blur frame1 dataset to help prevent the digit's identity from being smuggled into the position encoder
        blur_transform = torch.nn.Sequential(
            transforms.Resize(16),
            transforms.GaussianBlur(3),
            transforms.Resize(64)
        )

        # To save time at the cost of memory (I've got lots of the latter and not enough of the former),
        # I'll pre-transform the blurred data.
        self.blurred_frame = blur_transform(self.frame1.reshape(num_samples*10, 1, 64, 64)).reshape(num_samples, 10, 1, 64, 64)
        
    def __len__(self):
        return len(self.frame1)
        
    def __getitem__(self, index):
        return self.frame1[index], self.frame2[index], self.blurred_frame[index]    
    
# Returns an original and a blurred frame from the MovingMNIST dataset
class MovingMNISTBlurred(torch.utils.data.Dataset):
    def __init__(self, root, train, device, split=1000):
        super(MovingMNISTBlurred, self).__init__()
        mnist_test_seq = np.load(os.path.join(root, 'mnist_test_seq.npy')).swapaxes(0, 1).astype(np.float32)
        if train:
            self.data = torch.from_numpy(mnist_test_seq[:-split])
        else:
            self.data = torch.from_numpy(mnist_test_seq[-split:])

        self.data /= 255.0
        self.data = self.data.reshape(-1, 1, 64, 64)
        self.data = self.data.to(device)

        # Identity encoder is going to work with random pairs of images
        self.shuffle_pair_index()
        
        # Position encoder is going to work with digits whose identities have been blurred
        transform = torch.nn.Sequential(
            transforms.Resize(16),
            transforms.GaussianBlur(3),
            transforms.Resize(64)
        )

        # To save time at the cost of memory (I've got lots of the latter and not enough of the former),
        # I'll pre-transform the blurred data.
        self.blurred_data = transform(self.data)
        
    # Take the original data and, for each frame, reshuffle the frame it is paired with from the same video 
    def shuffle_pair_index(self):
        num_videos = (int)(len(self.data) / 20)

        # Frame index is 20 random choices 0-19, repeated for the number of videos 
        #frame_index = torch.randint(20, size=(num_videos * 20,))
        
        # Random choice is just too difficult for the model to learn anything
        # Let's just go with adjacent frame instead...
        frame_index = torch.tensor([(i%19)+1 for i in range(20)]).repeat(num_videos)

        # The video data has been flattened, so video 0 is the first 20 frames of the data,
        # video 1 is the next 20 frames, etc.
        video_index = (torch.tensor(range(num_videos))*20).repeat_interleave(20)

        # Adding the random frame_index with the sequential video_index gives us the index for the pair
        self.pair_index = frame_index + video_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.data[self.pair_index[index]], self.blurred_data[index]
    
# Encapsulates the latent embedding produced by the given encoder after running through "Moving MNIST"
class MovingMNISTEmbeddings(torch.utils.data.Dataset):
    def __init__(self, data_loader, encoder, batch_size, device):
        super(MovingMNISTEmbeddings, self).__init__()
        
        self.data = []
        encoder.eval()
        with torch.no_grad():
            for batch in data_loader:
                # 10x20x64x64 -> 200x1x64x64
                batch = torch.unsqueeze(torch.flatten(batch, 0, 1), 1)
                # 200x1x64x64 -> 200x128
                z = encoder(batch)
                # 200x128 -> 10x20x128 -- TODO: parameterize this better
                z = z.reshape(batch_size, 20, 128)
                self.data += z

        self.data = torch.stack(self.data).to(device)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return self.data[index]