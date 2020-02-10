from truecase_model import TrueCaser
from singlechar_dataset import TrueCaseDataset

import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torch import optim

from tqdm import trange, tqdm

TRAIN_PATH = 'data/wiki/input.txt'
VAL_PATH = 'data/wiki/val_input.txt'

HIDDEN_SIZE = 300

EPOCHS = 30
BATCH_SIZE = 100

def train(model, optimizer, criterion, train_loader, val_loader):
    for epoch_idx in range(1, EPOCHS + 1):
        # Epoch
        optimizer.zero_grad()

        loss_epoch = 0.
        batch_idx = 0
        num_batches = int(np.ceil(len(train_loader) / BATCH_SIZE))
        with tqdm(total=num_batches, leave=False) as pbar:
            for inputs, labels in train_loader:
                labels_predict, _ = model(inputs)

                labels = labels.squeeze()
                labels_predict = labels_predict.squeeze()
                loss_curr = criterion(labels_predict, labels)
                loss_curr.backward()

                loss_epoch += loss_curr.item()
                batch_idx += 1

                if batch_idx == BATCH_SIZE:
                    batch_idx = 0
                    optimizer.step()
                    pbar.update()
                    optimizer.zero_grad()

            if batch_idx != 0:
                batch_idx = 0
                optimizer.step()
                pbar.update()
                optimizer.zero_grad()

        loss_epoch /= len(train_loader)
        print('Epoch: {}\tLoss: {:.4f}'.format(batch_idx, loss_epoch))

    return model


def main(train_path, val_path):
    # Load data
    train_dataset = TrueCaseDataset(train_path)
    val_dataset = TrueCaseDataset(val_path, train_dataset.token_to_idx)

    train_loader = DataLoader(train_dataset)
    val_loader = DataLoader(val_dataset)

    # Load model
    model = TrueCaser(train_dataset.num_tokens(), HIDDEN_SIZE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Call train function
    model = train(model, optimizer, criterion, train_loader, val_loader)

    torch.save(model.state_dict(), 'trained_model.pth')


if __name__ == "__main__":
    main(TRAIN_PATH, VAL_PATH)
