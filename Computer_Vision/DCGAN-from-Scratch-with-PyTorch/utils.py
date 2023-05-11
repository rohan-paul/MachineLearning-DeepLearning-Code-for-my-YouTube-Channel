import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import math
import itertools
from glob import glob


def get_data_loader(batch_size):
    """
    This function creates a PyTorch DataLoader for the MNIST dataset.

    Parameters
    ----------
    batch_size : int
        The batch size to be used in the DataLoader.

    Returns
    -------
    DataLoader
        A DataLoader for the MNIST dataset.

    Note
    ----
    The DataLoader will shuffle and batch the data.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
    )

    train_dataset = datasets.MNIST(
        root="/content/drive/MyDrive/All_Datasets/MNIST",
        train=True,
        transform=transform,
        download=True,
    )

    # Data Loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    return train_loader


def plot_images(epoch, path, num_test_samples, generator, device):
    """
    This function generates a set of images using the provided generator model, and plots
    them in a grid structure.

    Parameters
    ----------
    epoch : int
        The current training epoch. Used for labeling the plot.
    path : str
        The path where the images will be saved.
    num_test_samples : int
        The number of images to generate and plot.
    generator : torch.nn.Module
        The generator model to use for creating the images.
    device : torch.device
        The device (CPU or GPU) where the generator model is located.

    Returns
    -------
    None
    """
    # Generate a random noise tensor with shape (num_test_samples, 100, 1, 1)
    # which will be used as input to the generator model to create images.
    z = torch.randn(num_test_samples, 100, 1, 1, device=device)

    # Calculate the size of the grid to plot based on the number of test samples.
    plot_grid_size = int(math.sqrt(num_test_samples))

    # Generate the images using the generator model.
    generated_fake_images = generator(z)

    # Append "variable_noise/" to the save path.
    path += "variable_noise/"

    # Initialize the title of the plot.
    title = "Variable Noise"

    # Create a figure with a grid of subplots.
    fig, ax = plt.subplots(plot_grid_size, plot_grid_size, figsize=(6, 6))

    # Hide the x and y axes of all subplots.
    for i, j in itertools.product(range(plot_grid_size), range(plot_grid_size)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    # Plot each of the generated images in its own subplot.
    for sample in range(num_test_samples):
        i = sample // 4
        j = sample % 4
        ax[i, j].cla()
        ax[i, j].imshow(
            generated_fake_images[sample].data.cpu().numpy().reshape(28, 28),
            cmap="Greys",
        )

    # Add a label for the epoch at the bottom of the plot.
    label = "Epoch_{}".format(epoch + 1)
    fig.text(0.5, 0.04, label, ha="center")

    # Add the title to the plot.
    fig.suptitle(title)
