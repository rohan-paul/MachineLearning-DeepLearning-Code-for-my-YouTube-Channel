import os
import numpy as np

import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import glob
import random
from torch.utils.data import Dataset
from PIL import Image


########################################################
# Methods for Image DataLoader
########################################################


def convert_to_rgb(image):
    """
    Convert an image to RGB format.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        PIL.Image.Image: RGB converted image.

    """
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "%sA" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%sB" % mode) + "/*.*"))
        # print("self.files_B ", self.files_B)
        """ Will print below array with all file names
        ['/content/drive/MyDrive/All_Datasets/summer2winter_yosemite/trainB/2005-06-26 14:04:52.jpg',
        '/content/drive/MyDrive/All_Datasets/summer2winter_yosemite/trainB/2005-08-02 09:19:52.jpg',..]
        """

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])
        # a % b => a is divided by b, and the remainder of that division is returned.

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = convert_to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = convert_to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        # Finally ruturn a dict
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


########################################################
# Replay Buffer
########################################################

"""
As per the paper -  To reduce model oscillation, we update the discriminator using a history of generated images rather than the
ones produced by the latest generators. We keep an image
buffer that stores the 50 previously created images.

And here the followed - https://arxiv.org/pdf/1612.07828.pdf


This is another strategy used to stabilize the CycleGAN Training

Replay buffer is used to train the discriminator. Generated images are added to the replay buffer and sampled from it.

The replay buffer returns the newly added image with a probability of 0.5.

Otherwise, it sends an older generated image and replaces the older image with the newly generated image.

This is done to reduce model oscillation. """


class ReplayBuffer:
    """
    A class used to represent a Replay Buffer. This buffer stores a certain number of previously
    generated images for training a Generative Adversarial Network (CycleGAN).
    This ReplayBuffer class is a useful tool in the context of CycleGAN as it can help
    to stabilize the learning process by reusing older generated images.

    ...

    Attributes
    ----------
    max_size : int
        maximum number of images that can be stored in the buffer
    data : list
        the list storing the images

    Methods
    -------
    push_and_pop(data):
        Adds new images to the buffer and returns a mixed batch of old and new images.
    """

    # We keep an image buffer that stores
    # the 50 previously created images.
    def __init__(self, max_size=50):
        """
        Constructs the necessary attributes for the ReplayBuffer object.

        Parameters
        ----------
            max_size : int
                maximum number of images that can be stored in the buffer. Should be greater than 0.
        """
        assert max_size > 0, "Empty buffer."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        """
        This method accepts a batch of images, saves them to the buffer, and returns a new batch of images.
        The returned batch is composed of some of the new images and, when the buffer is full, possibly some older images.

        Parameters
        ----------
            data : torch.Tensor
                The new images to add to the buffer.

        Returns
        -------
            torch.Tensor
                A batch of images consisting of new images and possibly some older images from the buffer.
        """
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                # If the buffer is full, decide whether to replace an old image in the buffer with the new one,
                # and whether to add the new image or an old image to the return batch.
                if random.uniform(0, 1) > 0.5:
                    # With a 50% chance, replace an old image in the buffer with the new image
                    # and add the old image to the return batch.
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[
                        i
                    ] = element  # replaces the older image with the newly generated image.
                else:
                    # With a 50% chance, keep the buffer as is and add the new image to the return batch.
                    to_return.append(element)
        return Variable(torch.cat(to_return))


########################################################
# Learning Rate scheduling with `lr_lambda`
########################################################


class LambdaLR:
    """
    A class used to represent a Learning Rate Scheduler that follows a LambdaLR policy.
    The learning rate decreases linearly after a specified epoch.

    ...

    Attributes
    ----------
    n_epochs : int
        total number of epochs for training
    offset : int
        number of epochs offset to be applied
    decay_start_epoch : int
        epoch from which learning rate decay should start

    Methods
    -------
    step(epoch):
        Calculates the multiplicative factor for the learning rate based on the current epoch.
    """
    def __init__(self, n_epochs, offset, decay_start_epoch):
        """
        Constructs the necessary attributes for the LambdaLR object.

        Parameters
        ----------
        n_epochs : int
            Total number of epochs for training.
        offset : int
            Number of epochs offset to be applied.
        decay_start_epoch : int
            Epoch from which learning rate decay should start.
        """
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        """
        This method calculates the multiplicative factor for the learning rate based on the current epoch.
        The learning rate decreases linearly after decay_start_epoch.

        Parameters
        ----------
        epoch : int
            The current training epoch.

        Returns
        -------
        float
            A multiplicative factor (between 1.0 and 0.0) for the learning rate.
        """
        # Below line checks whether the current epoch has exceeded the decay epoch(which is 100)
        # e.g. if current epoch is 80 then max (0, 80 - 100) will be 0.
        # i.e. then entire numerator will be 0 - so 1 - 0 is 1
        # i.e. the original LR remains as it is.
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )


########################################################
# Initialize convolution layer weights to N(0,0.02)
########################################################


def initialize_conv_weights_normal(m):
    """
    Initializes the weights and biases of Convolutional and Batch Normalization layers
    in a neural network model using normal distribution.

    Parameters
    ----------
    m : torch.nn.Module
        The module or layer in a PyTorch model, which is to be initialized.

    Returns
    -------
    None
    """
    # Extract the class name of the module to determine its type.
    classname = m.__class__.__name__

    # Check if the module is a Convolutional layer.
    if classname.find("Conv") != -1:
        # Initialize the weights of the Convolutional layer using a normal distribution
        # with mean 0.0 and standard deviation 0.02.
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

        # Check if the Convolutional layer has a bias attribute and is not None.
        if hasattr(m, "bias") and m.bias is not None:
            # Initialize the biases of the Convolutional layer as constant 0.
            torch.nn.init.constant_(m.bias.data, 0.0)

    # Check if the module is a Batch Normalization layer.
    elif classname.find("BatchNorm2d") != -1:
        # Initialize the weights of the Batch Normalization layer using a normal distribution
        # with mean 1.0 and standard deviation 0.02.
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)

        # Initialize the biases of the Batch Normalization layer as constant 0.
        torch.nn.init.constant_(m.bias.data, 0.0)