# relational_dataset.py

import numpy as np

class RelationalDataset:
    def __init__(self, n_samples=100, seq_len=4, seed=42):
        np.random.seed(seed)
        self.n_samples = n_samples
        self.seq_len = seq_len
        # generate random token angles between 0 and pi
        self.data = np.random.uniform(0, np.pi, size=(n_samples, seq_len))
        self.labels = np.zeros(n_samples)
        self.generate_labels()

    def generate_labels(self):
        # relational rule:
        # label = 1 if sin(first two tokens).sum() > sin(last two tokens).sum()
        for i in range(self.n_samples):
            seq = self.data[i]
            if np.sin(seq[:2]).sum() > np.sin(seq[-2:]).sum():
                self.labels[i] = 1
            else:
                self.labels[i] = 0

    def get_pair(self, index):
        # Return query and key as two sequences (for QRF)
        seq = self.data[index]
        label = self.labels[index]
        # for simplicity, define query = first half, key = second half
        mid = len(seq) // 2
        query_seq = seq[:mid]
        key_seq = seq[mid:]
        return query_seq, key_seq, label

    def __len__(self):
        return self.n_samples