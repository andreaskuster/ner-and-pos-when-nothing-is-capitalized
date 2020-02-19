from truecase import load_truecaser, load_truecase_dataset, evaluate
import pickle as pkl

import torch
from torch import nn
from torch.utils.data import DataLoader

# Get device for gpu support
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data is expected to be a file of one sentence per line.
# Look for datasets/wiki/test.txt for an example
dataset = load_truecase_dataset('datasets/wiki/test.txt')
data_loader = DataLoader(dataset, batch_size=100)

model = load_truecaser().to(device)
criterion = nn.CrossEntropyLoss()

loss_on_dataset, y_pred, y_truth = evaluate(model, criterion, data_loader)

# You can use y_pred/y_truth to get statistics about data
from sklearn.metrics import accuracy_score, f1_score
print(f'Loss: {loss_on_dataset}')
print('Accuracy: {:4f}'.format(accuracy_score(y_truth, y_pred)))
print('F1 Score: {:4f}'.format(f1_score(y_truth, y_pred)))
