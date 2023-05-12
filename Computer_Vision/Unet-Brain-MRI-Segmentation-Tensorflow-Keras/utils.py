import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import backend as K

plt.style.use("ggplot")


def plot_from_img_path(rows, columns, list_img_path, list_mask_path):
    """ This function plots a grid of images and their corresponding masks from given lists of image and mask paths.
    It uses matplotlib for plotting and OpenCV for reading and processing the images.

    Parameters:
    -----------
    rows : int
        The number of rows in the plot grid.

    columns : int
        The number of columns in the plot grid.

    list_img_path : list of str
        A list of paths to the image files to be plotted.

    list_mask_path : list of str
        A list of paths to the mask files to be plotted.

    Returns:
    --------
    None

    Notes:
    ------
    The images are displayed in RGB format, and the masks are overlaid on the images
    with a transparency of 0.4. """
    fig = plt.figure(figsize=(12, 12))
    for i in range(1, rows * columns + 1):
        fig.add_subplot(rows, columns, i)
        img_path = list_img_path[i]
        mask_path = list_mask_path[i]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        plt.imshow(image)
        plt.imshow(mask, alpha=0.4)
    plt.show()


def dice_coefficients(y_true, y_pred, smooth=100):
    """
    Computes the Dice similarity coefficient between the true and predicted values.

    The Dice coefficient is a statistical metric used to gauge the similarity of two samples.
    With a range from 0 to 1, the Dice coefficient is 1 when the two samples are identical and
    0 when they share no elements. A smoothing term is included to prevent division by zero.

    This function is typically used as a loss function for binary segmentation tasks,
    where the true and predicted values are binary masks of the same size.

    Parameters:
    -----------
    y_true : tensor
        The ground truth values. Typically a binary mask.

    y_pred : tensor
        The predicted values. Typically a binary mask.

    smooth : float, optional
        A smoothing factor to prevent division by zero. Default is 100.

    Returns:
    --------
    float
        The Dice similarity coefficient between the true and predicted values.

    Note:
    -----
    The inputs are flattened to 1D tensors before computation to handle
    both single and multi-channel inputs.
    """

    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)


def dice_coefficients_loss(y_true, y_pred, smooth=100):
    return -dice_coefficients(y_true, y_pred, smooth)


def iou(y_true, y_pred, smooth=100):
    """
    Calculates the Intersection over Union (IoU) between the true and predicted values.

    IoU, also known as the Jaccard Index, is a metric used to quantify the percent overlap
    between the target mask and our prediction output. It's often used in segmentation problems
    to evaluate the quality of predictions.

    This function is generally used for evaluating segmentation tasks where the true and
    predicted outputs are binary masks of the same size.

    Parameters:
    -----------
    y_true : tensor
        The ground truth values. Typically a binary mask.

    y_pred : tensor
        The predicted values. Typically a binary mask.

    smooth : float, optional
        A smoothing factor to prevent division by zero. Default is 100.

    Returns:
    --------
    float
        The Intersection over Union (IoU) between the true and predicted values.

    Note:
    -----
    The inputs are not flattened to 1D tensors before computation because Keras backend
    operations automatically broadcast the tensors to the appropriate shapes.
    """
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou


def jaccard_distance(y_true, y_pred):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    return -iou(y_true_flatten, y_pred_flatten)
