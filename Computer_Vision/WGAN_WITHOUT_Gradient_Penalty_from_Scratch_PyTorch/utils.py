from torchvision.utils import make_grid
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm

plt.ion()

def weights_init_normal(m):
    """
    Function to initialize the weights of the neural network layers with normal distribution.

    Parameters:
    - m (torch.nn.Module): The module to initialize the weights for.

    Returns:
    None
    """
    classname = m.__class__.__name__

    # Initialize Convolutional layer weights with normal distribution (mean=0, std=0.02)
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    # Initialize Batch Normalization layer weights with normal distribution (mean=1, std=0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



def to_img(x):
    x = x.clamp(0, 1)
    return x


########################################################
# Methods for Image Visualization
########################################################
def visualise_output(images, x, y):
    with torch.no_grad():
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = make_grid(images, x, y).numpy()
        figure(figsize=(20, 20))
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()


import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def visualise_output(images, x, y):
    """
    Function to visualize a grid of images.

    Parameters:
    - images (torch.Tensor): Tensor containing the images to visualize.
    - x (int): Number of images per row in the grid.
    - y (int): Number of images per column in the grid.

    Returns:
    None
    """
    with torch.no_grad():
        # Move the images to CPU if they are on a GPU device
        images = images.cpu()

        # Convert the images to the correct format for visualization
        images = to_img(images)

        # Convert the tensor grid to a numpy array
        np_imagegrid = make_grid(images, x, y).numpy()

        # Set the figure size for the plot
        plt.figure(figsize=(20, 20))

        # Transpose the image grid to the correct format and display it
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()
