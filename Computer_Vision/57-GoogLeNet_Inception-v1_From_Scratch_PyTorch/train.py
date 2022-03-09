import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_model(model, train_loader, val_loader, criterion, optimizer):
    EPOCHS = 15
    train_samples_num = 45000
    val_samples_num = 5000

    train_epoch_loss_history, val_epoch_loss_history = [], []

    for epoch in range(EPOCHS):

        train_running_loss = 0
        correct_train = 0

        model.train().cuda()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            """ for every mini-batch during the training phase, we
            typically want to explicitly set the gradients
            to zero before starting to do backpropragation """
            optimizer.zero_grad()

            # Start the forward pass
            prediction0, aux_pred_1, aux_pred_2 = model(inputs)

            # Compute the loss.
            real_loss = criterion(prediction0, labels)
            aux_loss_1 = criterion(aux_pred_1, labels)
            aux_loss_2 = criterion(aux_pred_2, labels)

            loss = real_loss + 0.3 * aux_loss_1 + 0.3 * aux_loss_2

            # do backpropagation and update weights with step()# Backward pass.
            loss.backward()
            optimizer.step()

            # Update the running corrects
            _, predicted = torch.max(prediction0.data, 1)

            correct_train += (predicted == labels).float().sum().item()

            """ Compute batch loss
            multiply each average batch loss with batch-length.
            The batch-length is inputs.size(0) which gives the number total images in each batch.
            Essentially I am un-averaging the previously calculated Loss """
            train_running_loss += loss.data.item() * inputs.shape[0]

        train_epoch_loss = train_running_loss / train_samples_num

        train_epoch_loss_history.append(train_epoch_loss)

        train_acc = correct_train / train_samples_num

        val_loss = 0
        correct_val = 0

        model.eval().cuda()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass.
                prediction0, aux_pred_1, aux_pred_2 = model(inputs)

                # Compute the loss.
                real_loss = criterion(prediction0, labels)
                aux_loss_1 = criterion(aux_pred_1, labels)
                aux_loss_2 = criterion(aux_pred_2, labels)

                loss = real_loss + 0.3 * aux_loss_1 + 0.3 * aux_loss_2

                # Compute training accuracy.
                _, predicted = torch.max(prediction0.data, 1)
                correct_val += (predicted == labels).float().sum().item()

                # Compute batch loss.
                val_loss += loss.data.item() * inputs.shape[0]

            val_loss /= val_samples_num
            val_epoch_loss_history.append(val_loss)
            val_acc = correct_val / val_samples_num

        info = "[For Epoch {}/{}]: train-loss = {:0.5f} | train-acc = {:0.3f} | val-loss = {:0.5f} | val-acc = {:0.3f}"

        print(
            info.format(
                epoch + 1, EPOCHS, train_epoch_loss, train_acc, val_loss, val_acc
            )
        )

        torch.save(
            model.state_dict(), "/content/sample_data/checkpoint{}".format(epoch + 1)
        )

    torch.save(model.state_dict(), "/content/sample_data/googlenet_model")

    return train_epoch_loss_history, val_epoch_loss_history
