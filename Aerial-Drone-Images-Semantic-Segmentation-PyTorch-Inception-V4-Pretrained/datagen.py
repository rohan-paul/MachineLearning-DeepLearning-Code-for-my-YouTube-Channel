from re import I
import time
import os
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn.functional as F

from PIL import Image
import cv2
import albumentations as A
import segmentation_models_pytorch as smp
from os.path import isfile, join
from os import listdir


class DataGen(Dataset):
    def __init__(self, img_path, mask_path, X, mean, std, transform=None, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.mean = mean
        self.std = std
        self.transform = transform
        self.patches = patch

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path, self.X[idx] + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path, self.X[idx] + ".png" + cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug["image"])
            mask = aug["mask"]

        if self.transform is None:
            img = Image.fromarray(img)

        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()

        if self.patches:
            img, mask = self.get_img_patches(img, mask)

    def get_img_patches(self, img, mask):
        """split images into patches fo size (512, 768)"""
        kh, kw = 512, 768  # kernel size
        dh, dw = 512, 768  # strides

        img_patches = img.unfold(1, kh, dh).unfold(2, kw, dw)

        img_patches = img_patches.contiguous().view(3, -1, kh, kw)

        img_patches = img_patches.permute(1, 0, 2, 3)

        # do the same for masks
        mask_patches = mask.unfold(0, kh, dh).unfold(1, kw, dw)
        mask_patches = mask_patches.contiguous().view(-1, kh, kw)

        return img_patches, mask_patches


class TestDataGen(Dataset):
    """
    Custom dataset class for loading test data.

    Args:
        img_path (str): Path to the directory containing the input images.
        mask_path (str): Path to the directory containing the corresponding masks.
        X (list): List of file names (without extensions) of the images and masks.
        transform (callable, optional): Optional transformations to apply to the images and masks.

    Returns:
        img (PIL.Image): Input image.
        mask (torch.Tensor): Corresponding mask.

    """

    def __init__(self, img_path, mask_path, X, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.

        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retrieves the image and mask at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            img (PIL.Image): Input image.
            mask (torch.Tensor): Corresponding mask.

        """
        img = cv2.imread(self.img_path + self.X[idx] + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + ".png", cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug["image"])
            mask = aug["mask"]

        if self.transform is None:
            img = Image.fromarray(img)

        mask = torch.from_numpy(mask).long()

        return img, mask
