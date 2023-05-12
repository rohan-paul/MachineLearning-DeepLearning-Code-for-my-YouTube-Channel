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
    """
    The Dice loss function for image segmentation models.

    The Dice loss is a measure of the overlap between the prediction (y_pred)
    and the ground truth (y_true). It ranges from 0 to 1, where a Dice loss
    of 1 indicates perfect overlap (i.e., a perfect segmentation), while a Dice
    loss of 0 indicates no overlap.

    The 'smooth' parameter is a small constant added to the numerator and
    denominator of the Dice coefficient to avoid division by zero errors
    and to stabilize the training.

    Parameters:
    y_true (tf.Tensor): Ground truth. Tensor of the same shape as y_pred.
    y_pred (tf.Tensor): Model prediction. Tensor output from the model.
    smooth (float, optional): A smoothing constant to avoid division by zero errors. Default is 100.

    Returns:
    float: The computed Dice loss.

    Why the negative sign here i.e. -dice_coefficients

    most optimization algorithms are designed to minimize a function rather than maximize it. Therefore, to convert the maximization problem to a minimization problem, we take the negative of the Dice coefficient. As a result, when the Dice coefficient is high (which is good), the loss is low, and when the Dice coefficient is low (which is bad), the loss is high. This allows the model to use standard optimization techniques to find the best parameters.

    """

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

""" Why intersection = K.sum(y_true * y_pred) in above

The line intersection = K.sum(y_true * y_pred) is calculating the intersection of two sets, where the sets are represented as binary masks (for a segmentation problem). The intersection is basically the overlapping region of the two sets.

This is done by performing an element-wise multiplication between the true values (y_true) and the predicted values (y_pred). In the context of binary masks, this operation essentially counts the number of pixels where both the true and predicted masks are 1 (indicating a positive class).

This is because in a binary mask, a pixel value of 1 denotes the presence of the object of interest (in a segmentation task, for instance), and a pixel value of 0 denotes the background or absence of the object. Thus, when both y_true and y_pred are 1, it means that both the ground truth and the prediction agree that there is an object at that particular pixel location.

In the context of binary masks for a segmentation problem, the masks represent the region of interest in an image, where '1' denotes the presence of an object (or class) and '0' denotes the absence of that object (or background).

When we multiply these masks element-wise (y_true * y_pred), we are looking for places where both masks agree that there is an object of interest. If both y_true and y_pred are 1 at a given pixel, then the product is 1, indicating an intersection at that pixel. If either of them is 0 at a given pixel, then the product is 0, indicating no intersection.

By summing up all these products (K.sum(y_true * y_pred)), we are effectively counting the number of pixels where both the ground truth (y_true) and the prediction (y_pred) agree that there is an object of interest. This is the intersection of the ground truth and prediction.


The K.sum() operation then sums up all these overlapping '1's to give a single number representing the total intersection, or overlap, between the true and predicted values.
"""


def jaccard_distance(y_true, y_pred):
    """
    Function to compute the Jaccard distance between the true labels and the predicted labels.

    The Jaccard distance, which is a measure of dissimilarity between sets, is computed as one minus
    the intersection over union (IoU) of the sets. A lower Jaccard distance between the predicted labels
    and the true labels indicates a better model fit.

    Parameters:
    y_true (np.array): The ground truth label array (binary mask).
    y_pred (np.array): The predicted label array (binary mask).

    Returns:
    float: The Jaccard distance between the ground truth labels and the predicted labels.
    """

    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    return -iou(y_true_flatten, y_pred_flatten)

""" Why flatten in jaccard_distance

In the context of image segmentation tasks, the ground truth y_true and the predicted output y_pred are typically 2D arrays (for grayscale images) or 3D arrays (for color images). Each element in these arrays corresponds to a pixel in the image, and the value of the element indicates the class or label of that pixel.

However, when calculating metrics such as the Jaccard distance or intersection over union (IoU), we're typically interested in comparing the sets of pixels that belong to each class, without considering their spatial arrangement in the image. Therefore, it's common to flatten the arrays to 1D before performing these calculations.

Flattening the arrays essentially transforms them into long lists of pixel values, disregarding the original spatial structure of the image. This allows us to treat the problem as a simple set comparison, where we're only interested in whether each pixel belongs to each class, not where the pixel is located in the image.

To summarize, flattening the arrays in this context simplifies the calculation of set-based metrics and focuses the evaluation on the pixel-wise accuracy of the segmentation, rather than the spatial accuracy. """