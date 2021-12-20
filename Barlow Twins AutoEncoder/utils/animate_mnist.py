import torch
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# Animates a pytorch tensor array of frames at the given framerate
def animate_mnist(video, framerate=10):
    
    def ToPILImage(frame):
        return transforms.ToPILImage()(frame)
        
    def animate(frame_idx):
        img.set_data(ToPILImage(video[frame_idx]))
        return img

    interval = 1000.0 / framerate
    
    # Need grayscale channel:
    video = torch.unsqueeze(video, 1)

    # Initialize video with the first frame
    fig = plt.figure()
    img = plt.imshow(ToPILImage(video[0]), cmap='gray', interpolation='none')
    plt.close()
    
    anim = animation.FuncAnimation(fig, animate, frames=video.shape[0], interval=interval)
    return HTML(anim.to_html5_video())

def plot_mnist(video):
    # Plot 20 frames of video, skipping alternating (to get better sense of motion)
    with torch.no_grad():
        frames = [transforms.ToPILImage()(video.reshape(20, 1, 64, 64)[i]) for i in range(0, 20, 2)]
        fig, axs = plt.subplots(1, 10, figsize=(20,5))
        axs = axs.flatten()
        for frame, ax in zip(frames, axs):
            ax.axis('off')
            ax.imshow(frame, cmap='gray')
        plt.show()