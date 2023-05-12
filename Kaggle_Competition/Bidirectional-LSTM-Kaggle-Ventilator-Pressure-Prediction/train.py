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

warnings.filterwarnings("ignore")
NUM_WORKERS = 4

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import get_linear_schedule_with_warmup

import random
import numpy as np

import time
import gc

import utils as utils

####################################################
################# VentilatorDataset  ###############
####################################################

class VentilatorDataset(Dataset):
    """
    Custom dataset class for the ventilator dataset.
    """
    def __init__(self, df):
        """
        Initialize the VentilatorDataset.

        Args:
            df (pandas.DataFrame): The input DataFrame containing the ventilator data.
        """
        if "pressure" not in df.columns:
            df['pressure'] = 0

        self.df = df.groupby('breath_id').agg(list).reset_index() #(75450, 8)
        """Above line first groupsby the "breath_id" column. This means that all rows with the same "breath_id" will be combined into a single group. The .agg(list) function aggregates the data in each group using the list function, which means that the values in each group will be combined into a list for each column. Something like this
        A       B             C

        X  [1, 3, 5]  [10, 30, 50]
        Y  [2, 4, 6]  [20, 40, 60]


        Then, .reset_index() is called to renumber the DataFrame's index and make it a new column, ensuring that the new DataFrame self.df has a continuous index starting from 0.
        """

        self.data_preprocess()

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.df.shape[0]

    def data_preprocess(self):
        """
        Preprocess the data and prepare input features for the PyTorch model.
        """
        self.pressures = np.array(self.df['pressure'].values.tolist())

        """  below block is responsible for preparing the input features for the PyTorch model. It reshapes and concatenates several NumPy arrays, calculates the cumulative sum of one of the arrays, and transposes the resulting concatenated array to match the expected input shape of the PyTorch model. """
        rs = np.array(self.df['R'].values.tolist()) # rs.shape => (75450, 80) # these shapes are for the whole dataframe i.e. before applying the below line
        # df_train = df_train[df_train['breath_id'] < 3]
        cs = np.array(self.df['C'].values.tolist()) # cs.shape => (75450, 80)
        u_ins = np.array(self.df['u_in'].values.tolist()) # u_ins.shape => (75450, 80)
        self.u_outs = np.array(self.df['u_out'].values.tolist()) # u_outs.shape => (75450, 80)

        self.inputs = np.concatenate([
            rs[:, None],
            cs[:, None],
            u_ins[:, None],
            np.cumsum(u_ins, 1)[:, None],
            self.u_outs[:, None]
        ], 1).transpose(0, 2, 1)

        """
        rs[:, None].shape => (75450, 1, 80)
        cs[:, None].shape => (75450, 1, 80)
        u_ins[:, None].shape => (75450, 1, 80)
        np.cumsum(u_ins, 1)[:, None].shape => (75450, 1, 80)
        self.u_outs[:, None].shape => (75450, 1, 80)

        .transpose(0, 2, 1) after the np.concatenate() will make the shape (75450, 80, 1)
        i.e. axis 0 remains in place, while it swaps axes 1 and 2.
        """

    def __getitem__(self, idx):
        """
        Get an item from the dataset at the specified index.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: A dictionary containing the input, u_out, and p tensors.
        """
        data = {
            "input": torch.tensor(self.inputs[idx], dtype=torch.float),
            "u_out": torch.tensor(self.u_outs[idx], dtype=torch.float),
            "p": torch.tensor(self.pressures[idx], dtype=torch.float),
        }

        return data


"""

### __getitem__() method

It allows the dataset object to be indexed using square brackets, like dataset[index], and it is used to fetch a single data point (sample) from the dataset given its index. The method is essential when creating custom PyTorch datasets to work seamlessly with PyTorch's DataLoader and other utilities.

 In summary, the __getitem__ method serves to fetch a single data point (sample) from the dataset given its index.

##### Explanation of the below block of code #############

self.inputs = np.concatenate([
        rs[:, None],
        cs[:, None],
        u_ins[:, None],
        np.cumsum(u_ins, 1)[:, None],
        self.u_outs[:, None]
    ], 1).transpose(0, 2, 1)

rs[:, None], cs[:, None], u_ins[:, None], self.u_outs[:, None]: Here, the NumPy arrays rs, cs, u_ins, and self.u_outs are being reshaped by adding a new axis. The [:, None] indexing operation adds a new axis to each array. This is done to facilitate concatenation along the new axis in the next step.

np.cumsum(u_ins, 1)[:, None]: This line calculates the cumulative sum of the u_ins array along axis 1 (columns). The cumulative sum is a new feature being created by summing the elements of u_ins along the specified axis. After calculating the cumulative sum, a new axis is added by using [:, None], similar to the previous step.

np.concatenate([...], 1): The np.concatenate function is used to concatenate the reshaped arrays along axis 1 (columns). The input is a list containing the reshaped arrays from the previous steps. The result is a new NumPy array with the same number of rows as the input arrays and columns corresponding to the concatenated arrays.

.transpose(0, 2, 1): calls the transpose function on the concatenated NumPy array and changes the order of the axes, specified by the input tuple (0, 2, 1). In this case, axis 0 remains in place, while it swaps axes 1 and 2. This operation adjusts the shape of the input array to match the expected input shape of the PyTorch model.

Reason for the above transpose() - In this case, the VentilatorDataset class is designed to work with a PyTorch model that expects input data in the shape of (batch_size, sequence_length, num_features). The transpose operation ensures that the input array conforms to this expected shape.
"""

#############################################################
################# Few Utils Methods for Training  ###########
#############################################################

class VentilatorLoss(nn.Module):
    """
    optimizes the competition metric
    """
    def __call__(self, preds, actual_pressure_vector, u_out):
        """
        Compute the loss between the predicted and actual pressure vectors.

        Args:
            preds (torch.Tensor): The predicted pressure vectors.
            actual_pressure_vector (torch.Tensor): The actual pressure vectors.
            u_out (torch.Tensor): The u_out values.

        Returns:
            torch.Tensor: The computed loss.

        In below line I am dropping out u_out=1 data, as is done by most of the top public LB notebooks which use weight=0 while computing the loss for samples at u_out=1.
        So below line will produce zero for u_out = 1 """
        weights = 1 - u_out
        mae = weights * (actual_pressure_vector - preds).abs()
        mae = mae.sum(-1) / weights.sum(-1)
        return mae

""" In above, the -1 in sum(-1) denotes the axis along which the summation should be performed. """


def count_parameters(model, all=False):
    """
    Counts the parameters of a model.

    Args:
        model (torch model): Model to count the parameters of.
        all (bool, optional):  Whether to count not trainable parameters. Defaults to False.

    Returns:
        int: Number of parameters.
    """
    return sum(p.numel() for p in model.parameters() if all or p.requires_grad)
    """
    In above the code basicaly is
    if all:
            return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    When called on a tensor object, numel() returns the product of the sizes of all dimensions of the tensor. For example, consider a tensor x with shape (2, 3, 4). The numel() method will return the value 24, which is equal to 2 x 3 x 4, the total number of elements in the tensor. """

def worker_init_fn(worker_id):
    """
     This function sets the seed for NumPy's random number generator using the calculated unique seed value for each worker.

     In Pytorch Dataloader - the worker_init_fn is an optional function that can be used to set up the worker's environment or seed when using multiple worker subprocesses to load data in parallel. It ensures that each worker operates with a different seed, helping maintain reproducibility in the training process.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)

""" np.random.get_state(): This function returns the internal state of NumPy's random number generator. The state is a tuple with multiple elements, where the second element (indexed by [1]) is an array representing the current seed value.

np.random.get_state()[1][0] + worker_id: This expression adds the worker_id to the current seed value. This ensures that each worker gets a unique seed for the random number generator, thus generating unique random sequences.

Generally, with np.random.seed() - We are controlling the seed value that initializes the internal state of the random number generator. From that state, the generator then produces a sequence of random numbers.
Seed values provide control over randomness since changing the seed gives you a different sequence. Using the same seed value on the same Random Number Generator will result in the exact same sequence of random numbers. T
"""


def save_model_weights(model, filename, verbose=1, cp_folder=""):
    """
    Saves the weights of a PyTorch model.

    Args:
        model (torch model): Model to save the weights of.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to save to. Defaults to "".
    """
    if verbose:
        print(f"\n -> Saving weights to {os.path.join(cp_folder, filename)}\n")
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


#############################################################
################# Fit method  ###############################
#############################################################
# Will use this fit() inside the train() method
def fit(
    model,
    train_dataset,
    val_dataset,
    loss_name="L1Loss",
    optimizer="Adam",
    epochs=200,
    batch_size=32,
    val_bs=32,
    warmup_prop=0.1,
    lr = 1e-3,
    num_classes=1,
    verbose=1,
    first_epoch_eval=0,
    device="cuda"):


    avg_val_loss = 0.

    optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=lr)
    """ getattr(torch.optim, optimizer): This function call retrieves a reference to the optimizer class specified by the optimizer string variable. For example, if optimizer is "Adam", getattr(torch.optim, optimizer) returns a reference to the torch.optim.Adam class.

    model.parameters(): This is a method call on the model object, which is an instance of a PyTorch neural network. The parameters() method returns an iterator over the model's trainable parameters (i.e., weights and biases). These parameters will be updated by the optimizer during the training process.

    getattr is a built-in Python function, not specific to PyTorch. It is used to retrieve the value of an attribute from an object by providing the object and the attribute's name as a string. It allows for dynamic attribute access, such as when you don't know the name of the attribute you want to access beforehand, or when you want to iterate over a list of attribute names.

    In PyTorch, getattr can be used to access attributes of various objects, like models, layers, or tensors.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    """ the worker_init_fn is an optional function that can be used to set up the worker's environment or seed when using multiple worker subprocesses to load data in parallel. It ensures that each worker operates with a different seed, helping maintain reproducibility in the training process. """

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    loss_func = VentilatorLoss()

    # Scheduler
    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = int(epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    # start training for each epoch
    for epoch in range(epochs):
        model.train()
        model.zero_grad()
        start_time = time.time()

        avg_loss = 0
        for data in train_loader:
            pred = model(data['input'].to(device)).squeeze(-1)

            loss = loss_func(
                pred,
                data['p'].to(device),
                data['u_out'].to(device),
            ).mean()
            loss.backward()
            avg_loss += loss.item() / len(train_loader)

            optimizer.step()
            scheduler.step()

            for param in model.parameters():
                param.grad = None

        model.eval()
        mae, avg_val_loss = 0, 0
        preds = []

        with torch.no_grad():
            for data in val_loader:
                pred = model(data['input'].to(device)).squeeze(-1)

                loss = loss_func(
                    pred.detach(),
                    data['p'].to(device),
                    data['u_out'].to(device),
                ).mean()
                avg_val_loss += loss.item() / len(val_loader)

                preds.append(pred.detach().cpu().numpy())

        preds = np.concatenate(preds, 0)
        mae = utils.mae_calculation(val_dataset.df, preds)

        elapsed_time = time.time() - start_time
        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} \t lr={lr:.1e}\t t={elapsed_time:.0f}s \t"
                f"loss={avg_loss:.3f}",
                end="\t",
            )

            if (epoch + 1 >= first_epoch_eval) or (epoch + 1 == epochs):
                print(f"val_loss={avg_val_loss:.3f}\tmae={mae:.3f}")
            else:
                print("")

    del (val_loader, train_loader, loss, data, pred)
    gc.collect()
    torch.cuda.empty_cache()

    return preds

#############################################################
################# LSTMModel ###############################
#############################################################


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim=4,
        lstm_dim=256,
        dense_dim=256,
        logit_dim=256,
        num_classes=1,
    ):
        """
        Initialize the LSTMModel.

        Args:
            input_dim (int): The input dimension.
            lstm_dim (int): The dimension of the LSTM hidden states.
            dense_dim (int): The dimension of the dense layers.
            logit_dim (int): The dimension of the logit layers.
            num_classes (int): The number of output classes.
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dense_dim // 2),
            nn.ReLU(),
            nn.Linear(dense_dim // 2, dense_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(dense_dim, lstm_dim, batch_first=True, bidirectional=True)

        self.logits = nn.Sequential(
            nn.Linear(lstm_dim * 2, logit_dim),
            nn.ReLU(),
            nn.Linear(logit_dim, num_classes),
        )

    def forward(self, x):
        """
        Forward pass of the LSTMModel.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The predicted output tensor.
        """
        features = self.mlp(x)
        features, _ = self.lstm(features)
        pred = self.logits(features)
        return pred

#############################################################
################# Train Mehod ###############################
#############################################################

def train(config, df_train, df_val, df_test, fold):
    """
    Trains and validate a model.

    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        df_test (pandas dataframe): Test metadata.
        fold (int): Selected fold.

    Returns:
        np array: Study validation predictions.
    """

    utils.seed_everything(config.seed)

    model = LSTMModel(
        input_dim=config.input_dim,
        lstm_dim=config.lstm_dim,
        dense_dim=config.dense_dim,
        logit_dim=config.logit_dim,
        num_classes=config.num_classes,
    ).to(config.device)

    model.zero_grad()

    train_dataset = VentilatorDataset(df_train)
    val_dataset = VentilatorDataset(df_val)
    test_dataset = VentilatorDataset(df_test)

    n_parameters = count_parameters(model)

    print(f"    -> {len(train_dataset)} training breathes")
    print(f"    -> {len(val_dataset)} validation breathes")
    print(f"    -> {n_parameters} trainable parameters\n")

    pred_val = fit(
        model,
        train_dataset,
        val_dataset,
        loss_name=config.loss,
        optimizer=config.optimizer,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
        verbose=config.verbose,
        first_epoch_eval=config.first_epoch_eval,
        device=config.device,
    )

    pred_test = predict(
        model,
        test_dataset,
        batch_size=config.val_bs,
        device=config.device
    )

    if config.save_weights:
        save_model_weights(
            model,
            f"{config.selected_model}_{fold}.pt",
            cp_folder="",
        )

    del (model, train_dataset, val_dataset, test_dataset)
    gc.collect()
    torch.cuda.empty_cache()

    return pred_val, pred_test

#########################################
################# predict() #############
#########################################


def predict(
    model,
    dataset,
    batch_size=64,
    device="cuda"
):
    """
    Usual torch predict function. Supports sigmoid and softmax activations.
    Args:
        model (torch model): Model to predict with.
        dataset (PathologyDataset): Dataset to predict on.
        batch_size (int, optional): Batch size. Defaults to 64.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(dataset) x num_classes]: Predictions.
    """
    model.eval()

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    preds = []
    with torch.no_grad():
        for data in loader:
            pred = model(data['input'].to(device)).squeeze(-1)
            preds.append(pred.detach().cpu().numpy())

    preds = np.concatenate(preds, 0)
    return preds