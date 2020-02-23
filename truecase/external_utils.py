from .singlechar_dataset import load_truecase_dataset
from .truecase_model import load_truecaser

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

def predict_truecasing(x, batch_size=100, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load deataset and model
    model = load_truecaser().to(device)
    dataset = load_truecase_dataset(x)

    # Get boolean array of whether sentence is 
    upper_idxs = __evaluate(model, DataLoader(dataset, batch_size=batch_size), device)

    # Apply casing on sentences
    result = []
    for casing, words in zip(upper_idxs, x):
        sentence_result = ''
        for is_up, character in zip(casing, ' '.join(words)):
            sentence_result += character.upper() if is_up else character

        result.append(sentence_result.split())

    return result

def __evaluate(model, data_loader, device):
    y_pred = []
    with torch.no_grad():
        for inputs, _, masks in tqdm(data_loader, desc='Predicting casing', leave=False):
            inputs = inputs.to(device)
            masks = masks.to(device)
            labels_predict, _ = model(inputs)

            # Get number of characters per sentence
            sums = torch.sum(masks, dim=1)

            # Also apply mask to them though
            labels_predict = labels_predict.reshape((-1, labels_predict.shape[-1]))
            masks = masks.reshape(-1)

            labels_predict = labels_predict[masks]

            labels_predict = torch.argmax(labels_predict, dim=1).cpu()

            # Unravel indexing into sentences
            prev = 0
            for s in sums:
                y_pred.append(labels_predict[prev:prev + s].tolist())
                prev += s

    return y_pred
