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

from utils import *

###############################################################################
def test_convert_to_rgb():
    # Create a sample image with a single channel (grayscale)
    image = Image.fromarray(np.zeros((100, 100), dtype=np.uint8), mode='L')

    # Convert the image to RGB format
    rgb_image = convert_to_rgb(image)

    # Check the output type and mode
    assert isinstance(rgb_image, Image.Image)
    assert rgb_image.mode == "RGB"

    # Check the output size
    assert rgb_image.size == image.size

    print("Unit test passed!")

# Run the unit test
# test_convert_to_rgb()
