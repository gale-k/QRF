# dataset_api.py

from datasets.relbench_loader import load_relbench
from datasets.rdbench_loader import load_rdbench
from datasets.wikidbs_loader import load_wikidbs

def load_relational_dataset(source, name, n_features=4):
    """
    Unified API for relational dataset loading.
    Returns: X (NxF), y (N)
    """
    if source == "relbench":
        return load_relbench(name, n_features)

    if source == "rdbench":
        return load_rdbench(name, n_features)

    if source == "wikidbs":
        return load_wikidbs(name, n_features)

    raise ValueError(f"Unknown dataset source: {source}")
