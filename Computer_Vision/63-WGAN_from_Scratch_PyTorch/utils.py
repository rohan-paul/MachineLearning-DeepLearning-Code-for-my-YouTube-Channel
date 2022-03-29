from torchvision.utils import make_grid
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm

plt.ion()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
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
