from torch.utils.data import Dataset
import torch

import numpy as np

from typing import Dict


class TrueCaseDataset(Dataset):
    def __init__(self, path: str, token_dict: Dict[str, int] = None):
        super().__init__()

        data_set = []
        if token_dict is None:
            unique_tokens = []
        with open(path, mode='r') as f:
            for l in f.readlines():
                data_set.append(list(l))
                if token_dict is None:
                    unique_tokens = np.union1d(unique_tokens, list(l))

        self.data_set = data_set
        if token_dict is None:
            # Tokens should be only lower case
            unique_tokens = np.unique([x.lower() for x in unique_tokens])

            self.token_to_idx = dict(zip(unique_tokens, list(range(len(unique_tokens)))))
        else:
            self.token_to_idx = token_dict

        self.idx_to_token = {}
        for k, v in self.token_to_idx.items():
            self.idx_to_token[v] = k

    def __len__(self):
        return len(self.data_set)

    def num_tokens(self):
        return len(self.idx_to_token)

    def get_idx(self, token):
        return self.token_to_idx[token]

    def get_token(self, idx):
        return self.idx_to_token[idx]

    def __getitem__(self, index):
        # Get x and y as lists
        original = self.data_set[index]
        lower = [x.lower() for x in original]
        target = [x != y for x, y in zip(original, lower)]  # 0 = lower case, 1 = upper case

        # Convert to indexes
        lower = [self.token_to_idx[x] for x in lower]

        # Convert to torch arrays
        lower = torch.tensor(lower).long()
        target = torch.tensor(target).long()

        return lower, target
