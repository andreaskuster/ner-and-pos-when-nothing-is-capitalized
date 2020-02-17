from truecase_model import TrueCaser
from singlechar_dataset import TrueCaseDataset

import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torch import optim

from tqdm import trange, tqdm

import os

TRAIN_PATH = '../datasets/wiki/input.txt'
VAL_PATH = '../datasets/wiki/val_input.txt'
MODEL_SAVE_PATH = 'models/'

HIDDEN_SIZE = 300

EPOCHS = 30
BATCH_SIZE = 100
OOV_RATE = 0.005

# AUTO FLAGS
_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('DEVICE USED: {}'.format(_DEVICE))


def train(model, optimizer, criterion, train_loader, val_loader):
    train_losses = []
    val_losses = []

    for epoch_idx in range(1, EPOCHS + 1):
        # Epoch
        optimizer.zero_grad()

        loss_epoch = 0.
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch_idx}', leave=False):
            optimizer.zero_grad()

            inputs, labels = inputs.to(_DEVICE), labels.to(_DEVICE)
            labels_predict, _ = model(inputs)

            # Reshape truth and predictions to match input to typical criterions
            labels_predict = labels_predict.reshape((-1, labels_predict.shape[-1]))
            labels = labels.reshape(-1)

            loss_curr = criterion(labels_predict, labels)

            loss_curr.backward()
            optimizer.step()

            loss_epoch += loss_curr.cpu().item()

        val_loss = 0.
        with torch.no_grad():
            for inputs, labels, masks in tqdm(val_loader, desc=f'Validation Epoch {epoch_idx}', leave=False):
                inputs, labels = inputs.to(_DEVICE), labels.to(_DEVICE)
                masks = masks.to(_DEVICE)
                labels_predict, _ = model(inputs)

                # Reshape truth and predictions to match input to typical criterions
                labels_predict = labels_predict.reshape((-1, labels_predict.shape[-1]))
                masks = masks.reshape(-1)
                labels = labels.reshape(-1)

                # Also apply mask to them though
                labels_predict = labels_predict[masks]
                labels = labels[masks]

                loss_curr = criterion(labels_predict, labels)

                val_loss += loss_curr.cpu().item()

        loss_epoch /= len(train_loader)
        print('Epoch: {}'.format(epoch_idx))
        print('\tLoss: {:.4f}'.format(loss_epoch))
        print('\tValidation Loss: {:.4f}'.format(val_loss))

        # Save model for each epoch
        torch.save(model.state_dict(), f'{MODEL_SAVE_PATH}/model_epoch_{epoch_idx}.pth')

        # Update loss tracking arrays
        val_losses.append(val_loss)
        train_losses.append(loss_curr)

    # Save loss arrays
    np.save(f'{MODEL_SAVE_PATH}/val_losses.npy', val_losses)
    np.save(f'{MODEL_SAVE_PATH}/train_losses.npy', train_losses)

    return model


def main(train_path, val_path):
    # Load data
    train_dataset = TrueCaseDataset(
        train_path,
        train=True,
        OOV_rate=OOV_RATE
    )
    val_dataset = TrueCaseDataset(
        val_path,
        token_dict=train_dataset.token_to_idx,
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Load model
    model = TrueCaser(train_dataset.num_tokens(), HIDDEN_SIZE).to(_DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    # Call train function
    model = train(model, optimizer, criterion, train_loader, val_loader)


if __name__ == "__main__":
    main(TRAIN_PATH, VAL_PATH)
