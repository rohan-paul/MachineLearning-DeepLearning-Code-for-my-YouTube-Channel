import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
NUM_WORKERS = 4

from train import *


#########################################
#########################################
#########################################

def mae_calculation(df, preds):
    """
    The competition will be scored as the mean absolute error between the predicted and actual pressures during the inspiratory phase of each breath. The expiratory phase is not scored. The score is given by:  | X - Y |
    where
        X is the vector of predicted pressure and
        Y is the vector of actual pressures across all breaths in the test set.
    """

    actual_pressure_vector = np.array(df['pressure'].values.tolist())

    """ "u_out" is a binary variable representing whether the expioratory valve is open (1) or closed (0) to let air out. u_out=1 is the expiratory phase. In simple terms, expiratory phase, is "breath out" or in the case of an artificial lung, let the air out. The data for the expiratory phase is not helpful for the model. During an exhale, the ventilator pressure and the target pressure are nearly identical.
    Hence here I am dropping out u_out=1 data, as is done by most of the top public LB notebooks which use weight=0 while computing the loss for samples at u_out=1.
    So below line will produce zero for u_out = 1
    """
    weights = 1 - np.array(df['u_out'].values.tolist())

    assert actual_pressure_vector.shape == preds.shape and weights.shape == actual_pressure_vector.shape, (actual_pressure_vector.shape, preds.shape, weights.shape)

    """
    (actual_pressure_vector.shape, preds.shape, weights.shape): This is a tuple that contains the shapes of all three arrays. This tuple will be displayed in the error message if the assertion fails.
    """

    mae = weights * np.abs(actual_pressure_vector - preds)
    mae = mae.sum() / weights.sum()

    return mae

#########################################
#########################################
#########################################

def k_fold(config, df, df_test):
    """
    Performs a patient grouped k-fold cross validation.
    """

    pred_oof = np.zeros(len(df))
    preds_test = []

    group_kfold = GroupKFold(n_splits=config.k)
    splits = list(group_kfold.split(X=df, y=df, groups=df["breath_id"]))
    """ In this case, the groups are defined by the "breath_id" column of the input DataFrame.
    The group_kfold.split() method returns an iterator that yields tuples of train and validation indices for each fold based on the specified groups. By passing the iterator to the list() function, a list of these tuples is created and stored in the splits variable.

    Each tuple in the splits list has the following structure:

    (train_indices, validation_indices)

    where train_indices and validation_indices are NumPy arrays containing the indices of the training and validation samples for a specific fold. This list has k elements, where k is the number of splits specified when creating the GroupKFold object.

    Read further in the official docs
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
    """

    """ Iterate over the generated splits using a for loop. For each split, if the current fold index (i) is in the config.selected_folds, perform the following steps:

   Create a training DataFrame (df_train) and a validation DataFrame (df_val) using the train and validation indices (train_idx and val_idx) from the current split.
     """
    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df_train = df.iloc[train_idx].copy().reset_index(drop=True)
            df_val = df.iloc[val_idx].copy().reset_index(drop=True)

            # Now get predicted val and test data by invoking the train()
            pred_val, pred_test = train(config, df_train, df_val, df_test, i)

            pred_oof[val_idx] = pred_val.flatten()
            preds_test.append(pred_test.flatten())

    print(f'\n -> CV MAE : {mae_calculation(df, pred_oof) :.3f}')

    return pred_oof, np.mean(preds_test, 0)

""" sklearn.model_selection.GroupKFold

K-fold iterator variant with non-overlapping groups.

Each group will appear exactly once in the test set across all folds (the number of distinct groups has to be at least equal to the number of folds).

The folds are approximately balanced in the sense that the number of distinct groups is approximately the same in each fold.

Returns => n_splits int > the number of splitting iterations in the cross-validator.

"""

#####################################################
#####################################################
#####################################################

def plot_prediction(sample_id, df):
    df_breath = df[df['breath_id'] == sample_id]

    cols = ['u_in', 'u_out', 'pressure'] if 'pressure' in df.columns else ['u_in', 'u_out']

    plt.figure(figsize=(12, 4))
    for col in cols:
        plt.plot(df_breath['time_step'], df_breath[col], label=col)

    metric = mae_calculation(df_breath, df_breath['pred'])

    plt.legend()
    plt.title(f'Sample {sample_id} - MAE={metric:.3f}')



#########################################
#########################################
#########################################

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
os.environ["PYTHONHASHSEED"] = str(seed): This line sets the seed for Python's hash function, which is used in creating hash values for various data types, like dictionaries. By setting the PYTHONHASHSEED environment variable, you ensure consistent hash values across different runs of the code, leading to consistent behavior when using hash-based data structures.

os.environ is a dictionary-like object in Python's os module, representing the environment variables of the operating system. It allows you to read, modify, and create new environment variables that will be inherited by child processes spawned by your Python script. One of these environment variables is PYTHONHASHSEED.

By default, Python uses a randomized hash function to prevent hash collision attacks (an attacker flooding a Python application with carefully crafted strings that cause hash collisions, leading to high CPU usage). However, this randomization can introduce non-deterministic behavior across different runs of the same script when using hash-based data structures like dictionaries and sets.

To ensure consistent behavior across different runs, you can set the PYTHONHASHSEED environment variable to a fixed integer value. When the variable is set, the Python interpreter will use the specified seed for its hash function, ensuring that the hash values for the same data are consistent across different runs of the script.

===============================

np.random.seed(seed): This line sets the seed for NumPy's random number generator

===============================

torch.manual_seed(seed): This line sets the seed for PyTorch's random number generator, used in various operations like weight initialization in neural networks. By setting the seed, you ensure that the initialization and random behavior in PyTorch remain consistent across different runs.

===============================

torch.cuda.manual_seed(seed): This line sets the seed for PyTorch's CUDA random number generator, which is responsible for generating random numbers on the GPU.

===============================

torch.backends.cudnn.deterministic = True: This line sets CuDNN (NVIDIA's Deep Neural Network library used by PyTorch) to use deterministic algorithms. Some CuDNN algorithms have non-deterministic behavior, which can lead to slight differences in the results for multiple runs. By enforcing deterministic algorithms, you ensure consistent behavior across runs when using CuDNN-accelerated functions.

===============================

torch.backends.cudnn.benchmark = False: This line disables the CuDNN benchmarking feature. When enabled, CuDNN automatically selects the best algorithm for a specific operation based on the input size and hardware. However, the best algorithm may change between runs, leading to non-deterministic behavior. Disabling the benchmarking feature ensures consistent behavior across runs at the expense of potentially suboptimal performance.
"""