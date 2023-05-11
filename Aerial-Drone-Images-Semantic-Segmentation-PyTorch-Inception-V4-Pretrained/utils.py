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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_classes = 23

def get_image_id_df(root_img_path):
    """
    Generate a DataFrame containing the image IDs.

    Args:
        root_img_path (str): Path to the directory containing the images.

    Returns:
        pd.DataFrame: A DataFrame containing the image IDs.

    """
    name = []
    filenames = [f for f in listdir(root_img_path) if isfile(join(root_img_path, f))]
    for filename in filenames:
        name.append(filename.split(".")[0])
    return pd.DataFrame({"id": name}, index=np.arange(0, len(name)))



def pixel_accuracy(predicted_image, mask):
    """
    Calculate the pixel accuracy between the predicted image and the ground truth mask.

    Args:
        predicted_image (torch.Tensor): Predicted image tensor of shape (N, C, H, W).
        mask (torch.Tensor): Ground truth mask tensor of shape (N, H, W).

    Returns:
        float: Pixel accuracy between the predicted image and the ground truth mask.

    """
    with torch.no_grad():
        # Convert predicted_image to class predictions
        predicted_image = torch.argmax(F.softmax(predicted_image, dim=1), dim=1)

        # Compare predicted_image with mask to get pixel-wise correctness
        correct = torch.eq(predicted_image, mask).int()

        # Calculate pixel accuracy
        accuracy = float(correct.sum()) / float(correct.numel())

    return accuracy


def mean_iou(predicted_label, label, eps=1e-10, num_classes=10):
    """
    Calculate the mean Intersection over Union (IoU) between the predicted labels and the ground truth labels.

    Args:
        predicted_label (torch.Tensor): Predicted label tensor of shape (N, C, H, W).
        label (torch.Tensor): Ground truth label tensor of shape (N, H, W).
        eps (float, optional): Epsilon value for numerical stability.
        num_classes (int, optional): Number of classes.

    Returns:
        float: Mean IoU value.

    """
    with torch.no_grad():
        # Convert predicted_label to class predictions
        predicted_label = F.softmax(predicted_label, dim=1)
        predicted_label = torch.argmax(predicted_label, dim=1)

        # Reshape predicted_label and label for easier computation
        predicted_label = predicted_label.contiguous().view(-1)
        label = label.contiguous().view(-1)

        iou_single_class = []
        for class_number in range(0, num_classes):
            true_predicted_class = predicted_label == class_number
            true_label = label == class_number

            if true_label.long().sum().item() == 0:
                iou_single_class.append(np.nan)
            else:
                # Calculate intersection and union
                intersection = (
                    torch.logical_and(true_predicted_class, true_label)
                    .sum()
                    .float()
                    .item()
                )
                union = (
                    torch.logical_or(true_predicted_class, true_label)
                    .sum()
                    .float()
                    .item()
                )

                # Calculate IoU for the current class
                iou = (intersection + eps) / (union + eps)
                iou_single_class.append(iou)

        # Calculate mean IoU across all classes
        return np.nanmean(iou_single_class)


#### Some Plotting Function ####

def plot_loss_vs_epoch(history):
    """
    Plot the training and validation loss versus epochs.

    Args:
        history (dict): Dictionary containing the training history with keys 'val_loss' and 'train_loss'.

    """
    plt.plot(history["val_loss"], label="val_loss", marker="o")
    plt.plot(history["train_loss"], label="Train loss", marker="o")
    plt.title("Loss per epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.grid()
    plt.show()

def plot_iou_score_vs_epoch(history):
    plt.plot(history["train_miou"], label="Train mIoU", marker="*")
    plt.plot(history["val_miou"], label="Val mIoU", marker="*")
    plt.title("mIoU Score per Epoch ")
    plt.ylabel("mean IoU")
    plt.xlabel("epoch")
    plt.show()


def plot_accuracy_vs_epoch(history):
    plt.plot(history["train_acc"], label="Train Accuracy", marker="*")
    plt.plot(history["val_acc"], label="Val Accuracy", marker="*")
    plt.title("Accuracy vs Epoch")
    plt.ylabel("Accuracy")
    plt.xlabel("epoch")
    plt.legend()
    plt.show()


def predict_image_mask_miou(
    model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)

        predicted_image = model(image)
        mean_iou_score = mean_iou(predicted_image, mask)
        masked = torch.argmax(predicted_image, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, mean_iou_score


def predict_iamge_mask_pixel_accuracy(
    model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    model.eval()
    t = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        predicted_image = model(image)
        pixel_accuracy = pixel_accuracy(predicted_image, mask)
        masked = torch.argmax(predicted_image, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, pixel_accuracy


def miou_score_from_trained_model(model, test_set):
    score_iou = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
    return score_iou


def pixel_accuracy_from_trained_model(model, test_set):
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, acc = predict_iamge_mask_pixel_accuracy(model, img, mask)
        accuracy.append(accuracy)
    return accuracy
