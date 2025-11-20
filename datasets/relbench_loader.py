# relbench_loader.py

from relbench.datasets import get_dataset
from relbench.tasks import get_task
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


# Load a RelBench dataset and task

def load_relbench(dataset_name: str, task_name: str, n_features: int = 4):
    """
    Loads a dataset and task from RelBench, extracts the main entity table,
    returns feature matrix X and target array y suitable for QRF attention.

    Parameters
    ----------
    dataset_name : str  (e.g. "rel-amazon", "rel-movielens")
    task_name    : str  (e.g. "user-churn", "product-category")
    n_features   : int  number of PCA dimensions for QRF encoding

    Returns
    -------
    X : np.ndarray  (N, n_features)   entity feature matrix
    y : np.ndarray  (N,)              entity labels for classification
    """

    print(f"[INFO] Loading dataset '{dataset_name}' task '{task_name}'...")

    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)

    # Get the main table used for the task (train split)
    table = task.get_table("train")        # RelBench Table object
    df = table.to_pandas()                 # Convert to pandas

    target_col = table.target_column       # The label column

    # Extract features and labels
    X_raw = df.drop(columns=[target_col]).select_dtypes(include=['float', 'int'])
    y = df[target_col].values

    # Convert to numpy
    X_raw = X_raw.values

    
    # Normalisation + PCA compression
    
    scaler = MinMaxScaler((0, 1))
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=n_features)
    X = pca.fit_transform(X_scaled)

    print(f"[INFO] Loaded {X.shape[0]} samples with {X.shape[1]} PCA features.")

    return X, y
