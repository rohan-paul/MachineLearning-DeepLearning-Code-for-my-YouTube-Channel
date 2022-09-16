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

"""
torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)

"""


class Generator(nn.Module):
    """
    noise_vector:  is the length of the z input vector.

    num_gen_filter: size of the feature maps that are propagated through the generator,

    num_ch: The number of channels in the output image (set to 1 for Grayscale images).

    Here, the height and width dimension of the image does not change, only the channel dimension decreases.

    For the Conv and ConvTranspose layers:
    * in_channels (int) – Number of channels/filters in the input image
    * out_channels (int) – Number of channels/filters produced by the convolution

    """

    def __init__(self, num_ch, noise_vector, num_gen_filter):
        super(Generator, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=noise_vector,
                out_channels=num_gen_filter * 4,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_gen_filter * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=num_gen_filter * 4,
                out_channels=num_gen_filter * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_gen_filter * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=num_gen_filter * 2,
                out_channels=num_gen_filter,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_gen_filter),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                in_channels=num_gen_filter,
                out_channels=num_ch,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

    def forward(self, input):
        output = self.network(input)
        return output


class Discriminator(nn.Module):
    """
    torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
    Here, the height and width dimension of the image does not change, only the channel dimension increases.
    """

    def __init__(self, num_ch, num_disc_filter):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=num_ch,
                out_channels=num_disc_filter,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=num_disc_filter,
                out_channels=num_disc_filter * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_disc_filter * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=num_disc_filter * 2,
                out_channels=num_disc_filter * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_disc_filter * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=num_disc_filter * 4,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    # The Discriminator outputs a scalar probability to classify the input image as real or fake.
    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(1)
