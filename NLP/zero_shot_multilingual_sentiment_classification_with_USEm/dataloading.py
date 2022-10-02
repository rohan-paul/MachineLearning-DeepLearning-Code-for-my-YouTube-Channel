import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, load_metric
import numpy as np

from typing import List, Dict

model_URL = 'https://tfhub.dev/google/universal-sentence-encoder-large/5'

encoder = hub.load(model_URL)


def embed_text(text: List[str]) -> List[np.ndarray]:
    vectors = encoder(text)
    return [vector.numpy() for vector in vectors]
''' Want the vectors to be numpy arrays, not Tensorflow tensors, b/c they'll be used in PyTorch. '''


def encoder_factory(label2int: Dict[str, int]):
    def encode(batch):
        batch["embedding"] = embed_text(batch["text"])
        batch["label"] = [label2int[str(x)] for x in batch["label"]]
        return batch

    return encode

class YelpDataLoader(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 32,
                 num_workers: int = 2):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = torch.cuda.is_available()  # type: ignore

    def prepare_data(self):
        ''' This method loads a subset of the train and test sets.
        It uses the first 2% of the train and test sets to
        train and test, respectively.
        It uses the last 1% of the training set for validation.
        '''
        self.test_ds = load_dataset('yelp_polarity', split="test[:2%]")
        self.train_ds = load_dataset('yelp_polarity', split="train[:2%]")
        self.val_ds = load_dataset('yelp_polarity', split="train[99%:]")

        self.label_names = self.train_ds.unique("label")
        label2int = {str(label): n for n, label in enumerate(self.label_names)}
        self.encoder = encoder_factory(label2int)

    def setup(self):
        # Compute embeddings in batches, so that they fit in the GPU's RAM.
        self.train = self.train_ds.map(self.encoder, batched=True, batch_size=self.batch_size)
        self.train.set_format(type="torch", columns=["embedding", "label"],                     # type: ignore
                              output_all_columns=True)

        self.val = self.val_ds.map(self.encoder, batched=True, batch_size=self.batch_size)
        self.val.set_format(type="torch", columns=["embedding", "label"],                       # type: ignore
                            output_all_columns=True)

        self.test = self.test_ds.map(self.encoder, batched=True, batch_size=self.batch_size)
        self.test.set_format(type="torch", columns=["embedding", "label"],                      # type: ignore
                             output_all_columns=True)

    def train_dataloader(self):
        return DataLoader(self.train,                                                           # type: ignore
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)


