import sys
sys.path.append('..')

from datasets.ptb.penn_treebank import PTB
from datasets.conll2003.conll2003 import CoNLL2003

from truecase import evaluate, load_truecaser, load_truecase_dataset

import torch
from sklearn.metrics import f1_score


# Load CoNLL2003
conll = CoNLL2003()
(train, _), (test, _) = conll.load_data()
train_dataset = load_truecase_dataset(train)
test_dataset = load_truecase_dataset(test)

# Load PTB
ptb = PTB()
ptb_1_18, _ = ptb.load_data(list(range(1, 19)))
ptb_22_24, _ = ptb.load_data(list(range(22, 25)))
ptb_1_18 = load_truecase_dataset(ptb_1_18)
ptb_22_24 = load_truecase_dataset(ptb_22_24)

# Load model, criterion
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_truecaser().to(device)
criterion = torch.nn.CrossEntropyLoss()

# Get all the results
for dataset, dataset_name in zip(
        [train_dataset, test_dataset, ptb_1_18, ptb_22_24],
        ['CoNLL Train', 'CoNLL Test', 'PTB 1-18', 'PTB 22-24']
):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=100)
    _, y_pred, y_truth = evaluate(model, criterion, data_loader)
    f1_result = f1_score(y_truth, y_pred)
    print(f'{dataset_name}: {f1_result}')
