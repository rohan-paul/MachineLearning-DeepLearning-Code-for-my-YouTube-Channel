import warnings

warnings.filterwarnings("ignore")

import numpy as np
import cv2

import tensorflow as tf

from config import *
from utils import *
import utils


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=BATCH_SIZE, subset="train", shuffle=False):
        super().__init__()
        self.df = df
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.indexes = np.arange(len(df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    """ __getitem__ returns a batch of images and masks """

    def __getitem__(self, index):
        X = np.empty((self.batch_size, 128, 128, 3))  # Makes a 4-D Tensor
        y = np.empty((self.batch_size, 128, 128, 3))  # Makes a 4-D Tensor

        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        for i, img_path in enumerate(self.df["path"].iloc[indexes]):
            # print("df['path'].iloc[indexes].shape ", self.df['path'].iloc[indexes].shape) # (16,)
            # in above 'i' is just the counter. i.e. starts from 0 and goes upto the max length of all the rows
            w = self.df["width"].iloc[
                indexes[i]
            ]  # selects the row number of indexes[i]
            h = self.df["height"].iloc[indexes[i]]
            img = self._load_grayscaled_img(img_path)  # shape: (128,128,1)
            # print('img shape after _load_grayscaled_img ', img.shape) #(128, 128, 1)
            # Now update X[i,] to be this image.
            X[
                i,
            ] = img  # broadcast to shape: (128,128,3)
            # As we know, that arr[1,] is equivalent to arr[1, :]
            # As NumPy will automatically insert trailing slices for you

            # print('X after ', X.shape) # (16, 128, 128, 3)
            # The slice notation in the above line means -
            # Set me the (i+1)th Row of X to be this image

            if self.subset == "train":
                for k, j in enumerate(["large_bowel", "small_bowel", "stomach"]):
                    # Now 'j' will take each value from the above list
                    # e.g. self.df['large_bowel']
                    # and in my train_df_rearranged each of the ["large_bowel","small_bowel","stomach"]
                    # column names contain RLE formatted segmentation data.
                    rles = self.df[j].iloc[indexes[i]]
                    # so the above line will actually be something like => self.df['stomach'].iloc[indexes[20]]
                    # giving me the RLE data for that row and column
                    # mask = rle_decode(rles, shape=(h, w, 1))
                    # if all my utils method is in separate file then uncomment below
                    mask = utils.rle_decode(rles, shape=(h, w, 1))
                    mask = cv2.resize(mask, (128, 128))
                    y[i, :, :, k] = mask
        if self.subset == "train":
            return X, y
        else:
            return X

    def _load_grayscaled_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
        img_size = (128, 128)
        img = cv2.resize(img, img_size)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=-1)
        return img

    """cv2.IMREAD_ANYDEPTH => If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit. """
