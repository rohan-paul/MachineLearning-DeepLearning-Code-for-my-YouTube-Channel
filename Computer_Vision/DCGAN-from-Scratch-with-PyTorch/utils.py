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
    # MNIST Dataset
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
    z = torch.randn(num_test_samples, 100, 1, 1, device=device)

    plot_grid_size = int(math.sqrt(num_test_samples))

    title = None

    generated_fake_images = generator(z)

    path += "variable_noise/"

    title = "Variable Noise"

    fig, ax = plt.subplots(plot_grid_size, plot_grid_size, figsize=(6, 6))

    for i, j in itertools.product(range(plot_grid_size), range(plot_grid_size)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for sample in range(num_test_samples):
        i = sample // 4
        j = sample % 4
        ax[i, j].cla()
        ax[i, j].imshow(
            generated_fake_images[sample].data.cpu().numpy().reshape(28, 28),
            cmap="Greys",
        )

    label = "Epoch_{}".format(epoch + 1)
    fig.text(0.5, 0.04, label, ha="center")
    fig.suptitle(title)
