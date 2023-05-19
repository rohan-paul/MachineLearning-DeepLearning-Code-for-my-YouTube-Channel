import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import cv2
import gc
from tqdm import tqdm
from datetime import datetime
from typing import Optional
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

from tensorflow import keras
import tensorflow as tf
import keras
from keras.models import load_model, save_model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.models import Model
from keras.layers import Input

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib as mpl

from config import *
import os

from train import *
import datagen


def df_preparation(df, subset="train", DEBUG=False):
    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
    df["slice"] = df["id"].apply(lambda x: x.split("_")[3])

    if (subset == "train") or (DEBUG):
        DIR = TRAIN_ROOT_DIR + "train"
    else:
        DIR = TEST_ROOT_DIR

    """Also another cool feature of `glob.glob`, if you want to avoid `/*/*/*` type of pattern( as not all datasets follows certain pattern hence we might not be able find how many `/` to use) you can use,

    glob.glob(`/kaggle/input/uw-madison-gi-tract-image-segmentation/train/**/*png', recursive=True) """
    all_images = glob(os.path.join(DIR, "**", "*.png"), recursive=True)
    print("all_images length ", len(all_images))  # 38496

    x = all_images[0].rsplit("/", 4)[0]
    # print('x ', x)
    # ../../input/uw-madison-gi-tract-image-segmentation/train

    # Now I need a column named 'path' holding the full path of all the images in this dataframe
    # But I can not simply create them with the below kind of line
    # df['path'] = all_images
    # Because each image is repeated. And so if I do the above line directly I will get below error
    # ValueError: Length of values (38496) does not match length of index (115488)
    # So the solution is to create a temporary dataframe > then merge this temp dataframe with the original df > then delete the temp df

    # To make a column which will have th full pathname of all the iamges
    # Hence I have to build the full path name. Below is an example.
    # '../../input/uw-madison-gi-tract-image-segmentation/train/case44/case44_day0/scans/slice_0085_266_266_1.50_1.50.png',
    path_partial_list = []
    for i in range(0, df.shape[0]):
        path_partial_list.append(
            os.path.join(
                x,
                "case" + str(df["case"].values[i]),
                "case"
                + str(df["case"].values[i])
                + "_"
                + "day"
                + str(df["day"].values[i]),
                "scans",
                "slice_" + str(df["slice"].values[i]),
            )
        )
    df["path_partial"] = path_partial_list

    # Now creating another temp df and for that, first I need the list of all images upto the string "slice_num"
    # print(str(all_images[4].rsplit("_",4)[0]))
    # ../../input/uw-madison-gi-tract-image-segmentation/train/case44/case44_day0/scans/slice_0088

    path_partial_list = []
    for i in range(0, len(all_images)):
        path_partial_list.append(str(all_images[i].rsplit("_", 4)[0]))

    tmp_df = pd.DataFrame()
    tmp_df["path_partial"] = path_partial_list
    tmp_df["path"] = all_images

    df = df.merge(tmp_df, on="path_partial").drop(columns=["path_partial"])

    df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
    df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))

    del x, path_partial_list, tmp_df

    return df


""" Rearrange Train Data Frame

In train.csv, for each id, we have 3 associated row representing the segmentation for the 3 classes. 'large_bowel', 'small_bowel', 'stomach' """


def df_rearrange_for_3_segmentation_classes(df, subset="train"):
    """
    Rearranges a DataFrame for three segmentation classes.

    This function takes a DataFrame `df` and rearranges it to have three segmentation classes: 'large_bowel',
    'small_bowel', and 'stomach'. The rearrangement is done based on the 'subset' parameter.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        subset (str, optional): The subset to rearrange. Defaults to 'train'.

    Returns:
        pandas.DataFrame: The rearranged DataFrame.

    """
    df_restructured = pd.DataFrame({"id": df["id"][::3]})

    if subset == "train":
        df_restructured["large_bowel"] = df["segmentation"][::3].values
        df_restructured["small_bowel"] = df["segmentation"][1::3].values
        df_restructured["stomach"] = df["segmentation"][2::3].values

    df_restructured["path"] = df["path"][::3].values
    df_restructured["case"] = df["case"][::3].values
    df_restructured["day"] = df["day"][::3].values
    df_restructured["slice"] = df["slice"][::3].values
    df_restructured["width"] = df["width"][::3].values
    df_restructured["height"] = df["height"][::3].values

    df_restructured = df_restructured.reset_index(drop=True)
    df_restructured = df_restructured.fillna("")
    if subset == "train":
        df_restructured["count"] = np.sum(
            df_restructured.iloc[:, 1:4] != "", axis=1
        ).values

    return df_restructured


# For the below rle_encode() method referring
# https://www.kaggle.com/code/paulorzp/rle-functions-run-lenght-encode-decode/script
def rle_encode(masked_image):
    """
    Encodes a masked image into a run-length encoded (RLE) string.

    This function takes a binary masked image represented as a numpy array and encodes it into a run-length
    encoded (RLE) string.

    Args:
        masked_image (numpy.ndarray): Binary masked image.

    Returns:
        str: Run-length encoded (RLE) string.

    """
    pixel = masked_image.flatten()

    pixel = np.concatenate([0], pixel, [0])

    # runs include indices to wherever 0s change to 1s or 1s change to 0s
    runs = np.where(pixel[1:] != pixel[:-1])[0] + 1

    runs[1::2] -= runs[::2]
    # runs[1::2] --> runs[start:stop:step], thus 2 here is the step
    # thus runs[1::2] includes the indices of the changing from 1 to 0

    # runs[::2] includes the indices for the changing from 0 to 1
    # runs[::2] includes the indices for the changing from 0 to 1
    return " ".join(str(x) for x in runs)


def rle_decode(mask_rle, shape, color=1):
    """
    Decodes a run-length encoded (RLE) string into an image mask.

    This function takes a run-length encoded (RLE) string, the desired shape of the output mask, and an optional color
    value. It decodes the RLE string into an image mask represented as a numpy array.

    Args:
        mask_rle (str): Run-length encoded (RLE) string.
        shape (tuple): Desired shape of the output mask.
        color (int, optional): Color value for the mask. Defaults to 1.

    Returns:
        numpy.ndarray: Decoded image mask.

    """
    s = mask_rle.split()
    starts, length = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + length
    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = color
    return img.reshape(shape)


def dice_coef(y_true, y_pred, smooth=1):
    """
    Calculates the Dice coefficient between two tensors.

    This function calculates the Dice coefficient, a commonly used metric for evaluating the similarity between
    two binary tensors or masks. The Dice coefficient is defined as the ratio of twice the intersection of the
    tensors to the sum of their sizes.

    Args:
        y_true (tensor): True binary tensor.
        y_pred (tensor): Predicted binary tensor.
        smooth (float, optional): Smoothing factor. Defaults to 1.

    Returns:
        float: Dice coefficient between the two tensors.

    """
    y_true_flattened = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flattened * y_pred_flatten)
    union = K.sum(y_true_flattened) + K.sum(y_pred_flatten)
    return (2.0 * intersection + smooth) / union + smooth


def dice_loss(y_true, y_pred):
    """
    Calculates the Dice loss between the true labels and predicted labels.

    The Dice loss measures the similarity between two sets by computing the Dice coefficient.
    It is commonly used in image segmentation tasks.

    Args:
        y_true (array-like): True labels or ground truth values.
        y_pred (array-like): Predicted labels.

    Returns:
        float: Dice loss value.
    """
    smooth = 1
    y_true_flattened = y_true.flatten()
    y_pred_flattened = y_pred.flatten()
    intersection = y_true_flattened * y_pred_flattened
    score = (2.0 * K.sum(intersection) + smooth) / (
        K.sum(y_true_flattened) + K.sum(y_pred_flattened) + smooth
    )
    return 1.0 - score


def iou_coef(y_true, y_pred, smooth):
    """
    Calculates the Intersection over Union (IoU) coefficient between the true labels and predicted labels.

    The IoU coefficient is a common evaluation metric for image segmentation tasks.
    It measures the overlap between two sets by computing the ratio of the intersection to the union of the sets.

    Args:
        y_true (array-like): True labels or ground truth values.
        y_pred (array-like): Predicted labels.
        smooth (float, optional): Smoothing parameter to avoid division by zero. Default is 1.

    Returns:
        float: IoU coefficient value.
    """
    intersection = K.sum(K.abs(y_true, *y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3] + K.sum(y_pred, [1, 2, 3])) - intersection
    iou = K.mean((intersection) + smooth / (union + smooth), axis=0)
    return iou


def bce_dice_loss(y_true, y_pred):
    """
    Calculates the binary cross-entropy (BCE) Dice loss between the true labels and predicted labels.

    The BCE Dice loss is a combination of the binary cross-entropy loss and the Dice loss.
    It is commonly used in image segmentation tasks to optimize for both accuracy and overlap.

    Args:
        y_true (array-like): True labels or ground truth values.
        y_pred (array-like): Predicted labels.

    Returns:
        float: BCE Dice loss value.
    """
    return binary_crossentropy(tf.cast(y_true, tf.float32), y_pred) + 0.5 * dice_loss(
        tf.cast(y_true, tf.float32), y_pred
    )


def plot_bar(df):
    plt.figure(figsize=(12, 6))
    bar = plt.bar([1, 2, 3], 100 * np.mean(df.iloc[:, 1:4] != "", axis=0))
    plt.title("Percent Training Images with Mask", fontsize=16)
    plt.ylabel("Percent of Train images with mask")
    plt.xlabel("Class Types")
    # labels = ["large bowel", "small bowel", "stomach"]
    labels = ["large_bowel", "small_bowel", "stomach"]

    for rect, lbl in zip(bar, labels):
        height = rect.get_height()
        plt.text(
            rect.get_x() + rect.get_width() / 3,
            height,
            lbl,
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.ylim((0, 50))
    plt.show()


def plot_mask_with_color_patches(df, colors, labels):
    list_indices_of_mask_random = list(
        df[df["large_bowel"] != ""].sample(BATCH_SIZE).index
    )
    list_indices_of_mask_random += list(
        df[df["small_bowel"] != ""].sample(BATCH_SIZE * 2).index
    )
    list_indices_of_mask_random += list(
        df[df["stomach"] != ""].sample(BATCH_SIZE * 3).index
    )
    # print('list_indices_of_mask_random ', list_indices_of_mask_random)
    # It will be a list of indexes like [15176, 13709, 30423, ..., 12730]

    batches_from_datagen = datagen.DataGenerator(
        df[df.index.isin(list_indices_of_mask_random)], shuffle=True
    )

    num_rows = 6

    fig = plt.figure(figsize=(10, 25))
    gs = gridspec.GridSpec(nrows=num_rows, ncols=2)
    patches = [
        mpatches.Patch(color=colors[i], label=f"{labels[i]}")
        for i in range(len(labels))
    ]

    cmap1 = mpl.colors.ListedColormap(colors[0])
    cmap2 = mpl.colors.ListedColormap(colors[1])
    cmap3 = mpl.colors.ListedColormap(colors[2])
    """ The `matplotlib.colors.ListedColormap` class is used to create colarmap objects from a list of colors.
    The class belongs to the `matplotlib.colors` module. This module is used for converting color or numbers arguments to RGBA or RGB and for mapping numbers to colors or color specification conversion in a 1-D array of colors also known as colormap.
    This can be useful for directly indexing into colormap and it can also be used to create special colormaps for normal mapping. """

    for i in range(num_rows):
        images, mask = batches_from_datagen[i]
        # print('images.shape ', images.shape) # (16, 128, 128, 3)
        # print('mask.shape ', mask.shape) # (16, 128, 128, 3)
        """
        For each ID, we are going to create an image of shape [img height, img width, 3], where 3 (number of channels) are the 3 layers for each class:

        * the first layer: large bowel
        * the second layer: small bowel
        * the third layer: stomach
        """
        sample_img = images[0, :, :, 0]  # After this the shapes will be (128, 128)
        mask1 = mask[0, :, :, 0]  # After this the shapes will be (128, 128)
        mask2 = mask[0, :, :, 1]  # After this the shapes will be (128, 128)
        mask3 = mask[0, :, :, 2]  # After this the shapes will be (128, 128)

        ax0 = fig.add_subplot(gs[i, 0])  # i here is the row-counter which is 6
        im = ax0.imshow(sample_img, cmap="bone")

        ax1 = fig.add_subplot(gs[i, 1])
        if i == 0:
            ax0.set_title("Image", fontsize=15, weight="bold", y=1.02)
            ax1.set_title("Mask", fontsize=15, weight="bold", y=1.02)
            plt.legend(
                handles=patches,
                bbox_to_anchor=(1.1, 0.65),
                loc=2,
                borderaxespad=0.4,
                fontsize=14,
                title="Mask Labels",
                title_fontsize=14,
                edgecolor="black",
                facecolor="#c5c6c7",
            )

        # print('mask1 ', mask1.shape) # (128, 128)
        # print('mask2 ', mask2.shape) # (128, 128)
        # print('mask3 ', mask3.shape) # (128, 128)
        # print('np.ma.masked_where(mask1== False,  mask1) ', np.ma.masked_where(mask1== True,  mask1))
        l0 = ax1.imshow(sample_img, cmap="bone")
        l1 = ax1.imshow(np.ma.masked_where(mask1 == False, mask1), cmap=cmap1, alpha=1)
        l2 = ax1.imshow(np.ma.masked_where(mask2 == False, mask2), cmap=cmap2, alpha=1)
        l3 = ax1.imshow(np.ma.masked_where(mask3 == False, mask3), cmap=cmap3, alpha=1)
        # l1 = ax1.imshow(np.ma.masked_where(mask1== 0,  mask1),cmap=cmap1, alpha=1)
        # l2 = ax1.imshow(np.ma.masked_where(mask2== 0,  mask2),cmap=cmap2, alpha=1)
        # l3 = ax1.imshow(np.ma.masked_where(mask3== 0,  mask3),cmap=cmap3, alpha=1)
        _ = [ax.set_axis_off() for ax in [ax0, ax1]]

        colors = [im.cmap(im.norm(1)) for im in [l1, l2, l3]]
