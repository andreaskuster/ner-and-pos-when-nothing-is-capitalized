import json

import numpy as np

tokens = []
labels = []

with open('f.json', mode='r') as f:
    for line in f.readlines():
        d = json.loads(line)
        tokens.append(d['tokens'])
        labels.append(d['entities'])

for token, label in zip(tokens, labels):
    assert(len(token) == len(label))  # Make sure there is a match


unique_labels = []
for l in labels:
    unique_labels.extend(l)
print('Unique labels: {}'.format(np.unique(unique_labels)))
print('')
print('Example sentence (tokens): {}'.format(tokens[0]))
print('Example sentence (labels): {}'.format(labels[0]))

np.save('tokens.npy', tokens, allow_pickle=True)
np.save('labels.npy', labels, allow_pickle=True)
