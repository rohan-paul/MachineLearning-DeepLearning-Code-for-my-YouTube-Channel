import imp
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

from utils import *
from datagen import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train(
    epochs,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    patch=False,
):
    torch.cuda.empty_cache()
    losses_train = []
    losses_test = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    decreases = 1
    num_of_times_loss_not_improving = 0

    model.to(device)
    fit_time = time.time()
    for epoch in range(epochs):
        start_time = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training Loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            image_tiles, mask_tiles = data
            if patch:
                batch_size, n_tiles, channel, height, width = image_tiles.size()
                image_tiles = image_tiles.view(-1, channel, height, width)
                mask_tiles = mask_tiles.view(-1, channel, height, width)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)

            # Forward Propagation
            predicted_image = model(image)
            loss = criterion(predicted_image, mask)

            # Metric to do Evaluation
            iou_score += mean_iou(predicted_image, mask)
            accuracy += pixel_accuracy(predicted_image, mask)

            # Backward Propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0

            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    image_tiles, mask_tiles = data

                    if patch:
                        batch_size, n_tiles, channel, height, width = image_tiles.size()
                        image_tiles = image_tiles.view(-1, channel, height, width)
                        mask_tiles = mask_tiles.view(-1, height, width)

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)

                    # Forward Propagation
                    predicted_image = model(image)

                    # Metric to do Evaluation
                    val_iou_score += mean_iou(predicted_image, mask)
                    test_accuracy += pixel_accuracy(predicted_image, mask)

                    loss = criterion(predicted_image, mask)
                    test_loss += loss.item()

            # Mean IoU for each batch calculation
            losses_train.append(running_loss / len(train_loader))
            losses_test.append(test_loss / len(val_loader))

            # Checking for Loss Decreases
            if min_loss > (test_loss / len(val_loader)):
                print(
                    "Loss Decreasing... {:.3f} >> {:.3f} ".format(
                        min_loss, (test_loss / len(val_loader))
                    )
                )
                min_loss = test_loss / len(val_loader)
                decreases += 1
                if decreases % 5 == 0:
                    print("Saving Model as loss is decreasing..")
                    torch.save(
                        model,
                        "Inception-v4_mIoU-{:.3f}.pt".format(
                            val_iou_score / len(val_loader)
                        ),
                    )

                # If the Loss is NOT decreasing
                if (test_loss / len(val_loader)) > min_loss:
                    min_loss = test_loss / len(val_loader)
                    print(
                        f"Loss Not Decreasing for {num_of_times_loss_not_improving} time"
                    )
                    if num_of_times_loss_not_improving == 6:
                        print(
                            "Loss not decreasing for 6 times, hence stopping Training"
                        )
                        break

                # Updating IoU and and Accuracy
                train_iou.append(iou_score / len(train_loader))
                train_acc.append(accuracy / len(train_loader))
                val_iou.append(val_iou_score / len(val_loader))
                val_acc.append(test_accuracy / len(val_loader))

                print(
                    "Epoch:{}/{}..".format(epoch + 1, epochs),
                    "Train Loss:{:.3f}..".format(running_loss / len(train_loader)),
                    "Validation Loss: {:.3f}..".format(test_loss / len(val_loader)),
                    "Train mean_iou:{:.3f}..".format(iou_score / len(train_loader)),
                    "Validation mean_iou: {:.3f}..".format(
                        val_iou_score / len(val_loader)
                    ),
                    "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                    "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                    "Time: {:.2f}m".format((time.time() - start_time) / 60),
                )

    history = {
        "train_loss": losses_train,
        "val_loss": losses_test,
        "train_miou": train_iou,
        "val_iou": val_iou,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "lrs": lrs,
    }
    print("Total time: {:.2f} m".format((time.time() - fit_time) / 60))
    return history
