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


BATCH_SIZE = 16
EPOCH = 10
n_splits = 5
fold_selected = 2  # 1,...,5


#### ONLY CHANGE DIRECTORIES HERE AND NO NEED TO CHANGE IT ANYWHERE ELSE IN THIS NOTEBOOK #####
# Kaggle Root Dir
# TRAIN_ROOT_DIR = "../input/uw-madison-gi-tract-image-segmentation/"
# TEST_ROOT_DIR = "../input/uw-madison-gi-tract-image-segmentation/test"

TRAIN_ROOT_DIR = "../../input/uw-madison-gi-tract-image-segmentation/"
TEST_ROOT_DIR = "../../input/uw-madison-gi-tract-image-segmentation/test/"
