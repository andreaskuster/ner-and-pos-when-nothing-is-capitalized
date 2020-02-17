from torch.utils.data import Dataset
import torch

import numpy as np

from typing import Dict


class TrueCaseDataset(Dataset):
    def __init__(self, path: str, token_dict: Dict[str, int] = None, train: bool = False, OOV_rate: float = 0.0):
        super().__init__()

        data_set = []
        max_len = -np.inf
        if token_dict is None:
            unique_tokens = []
        with open(path, mode='r') as f:
            for l in f.readlines():
                max_len = max(max_len, len(list(l)))
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

        # Add padding
        if 'P' not in self.token_to_idx:
            self.token_to_idx['P'] = len(self.token_to_idx)

        # Add OOV token
        if 'O' not in self.token_to_idx:
            self.token_to_idx['O'] = len(self.token_to_idx)

        # Add max length of sentence
        self.max_len = max_len

        self.idx_to_token = {}
        for k, v in self.token_to_idx.items():
            self.idx_to_token[v] = k

        self.OOV_rate = OOV_rate
        self.train = train

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

        if self.train:
            # Convert to indexes
            lower = [self.token_to_idx[x] for x in lower]

            # Add padding
            lower += [self.token_to_idx['P'] for _ in range(self.max_len - len(lower))]
            target += [0 for _ in range(self.max_len - len(target))]

            # Apply OOV
            lower = np.array(lower)
            idx = np.random.rand(len(lower)) <= self.OOV_rate
            lower[idx] = self.token_to_idx['O']

            # Convert to torch arrays
            lower = torch.tensor(lower).long()
            target = torch.tensor(target).long()

            return lower, target
        else:
            # Convert to indexes
            lower = [
                self.token_to_idx[x] if x in self.token_to_idx else self.token_to_idx['O']
                for x in lower
            ]

            # Add padding
            mask = [1] * len(lower) + [0] * (self.max_len - len(lower))
            lower += [self.token_to_idx['P'] for _ in range(self.max_len - len(lower))]
            target += [0 for _ in range(self.max_len - len(target))]

            # Convert to torch arrays
            lower = torch.tensor(lower).long()
            target = torch.tensor(target).long()
            mask = torch.tensor(mask, dtype=bool)

            return lower, target, mask
