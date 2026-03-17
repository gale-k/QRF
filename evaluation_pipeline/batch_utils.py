# batch_utils.py

import numpy as np

def sample_batch(dataset, batch_size):

    dataset_size = len(dataset)

    batch_size = min(batch_size, dataset_size)

    indices = np.random.choice(dataset_size, batch_size, replace=False)

    batch = []
    for i in indices:
        batch.append(dataset.get_pair(i))

    return batch

def sample_keys(dataset, num_keys):

    dataset_size = len(dataset)

    num_keys = min(num_keys, dataset_size)

    indices = np.random.choice(dataset_size, num_keys, replace=False)

    keys = []
    for i in indices:
        _, key_angle, _ = dataset.get_pair(i)
        keys.append(key_angle)

    return keys

