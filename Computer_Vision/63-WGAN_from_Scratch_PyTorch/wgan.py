import torch.nn as nn
import torch.nn.functional as F
import numpy as np

##############################################
# Generator
##############################################


class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim):
        super(Generator, self).__init__()

        def block(in_features, out_features, normalize=True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(
                in_features=latent_dim, out_features=128, normalize=False
            ),  # Batch_size, 784 -> Batch_size, 128
            *block(
                in_features=128, out_features=256
            ),  # Batch_size, 128 -> Batch_size, 256
            *block(
                in_features=256, out_features=512
            ),  # Batch_size, 256 -> Batch_size, 512
            *block(
                in_features=512, out_features=1024
            ),  # Batch_size, 512 -> Batch_size, 1024
            nn.Linear(
                in_features=1024, out_features=int(np.prod(img_shape))
            ),  # Batch_size, 1024 -> Batch_size, np.prod(img_shape)
            nn.Tanh()
        )

    def forward(self, img_shape, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


##############################
# Critic
##############################
class Critic(nn.Module):
    def __init__(self, img_shape):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(
                in_features=int(np.prod(img_shape)), out_features=512
            ),  # Batch_size, np.prod(img_shape) -> Batch_size, 512
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(
                in_features=512, out_features=256
            ),  # Batch_size, 512 -> Batch_size, 256
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(
                in_features=256, out_features=1
            ),  # Batch_size, 256 -> Batch_size, 1
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity
