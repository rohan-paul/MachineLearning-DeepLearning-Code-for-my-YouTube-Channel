import os
import torch

# ROOT_DIR = '../input/feedback-prize-2021/' # Kaggle
ROOT_DIR = '../input/' # local

# This is used to download the model from the huggingface hub
# MODEL_NAME = 'roberta-large'
MODEL_NAME = 'roberta-base'

# Path where to download the model
MODEL_PATH = 'model'

RUN_NAME = f"{MODEL_NAME}-4"

# Max length for the tokenization and the model
# For BERT-like models it's 512 in general
MAX_LEN = 512

# The authorized overlap between two part of the context when splitting it is needed.
# The overlapping tokens when chunking the texts
# Possibly a power of 2 would have been better
DOC_STRIDE = 128

# Training configuration
# 5 epochs with different learning rates (inherited from Chris')
# Haven't tried variations yet
config = {'train_batch_size': 1,
          'valid_batch_size': 1,
          'epochs': 1,
          'learning_rates': [1],
          'max_grad_norm': 10,
          'device': 'cuda' if torch.cuda.is_available() else 'cpu',
          'model_name': MODEL_NAME,
          'max_length': MAX_LEN,
          'doc_stride': DOC_STRIDE,
          }

# Note in above, I have 5 Learning rates for 5 epochs

output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim',
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

LABELS_TO_IDS = {v:k for k,v in enumerate(output_labels)}
IDS_TO_LABELS = {k:v for k,v in enumerate(output_labels)}