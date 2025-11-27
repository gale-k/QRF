# relbench_loader.py

from relbench.datasets import get_dataset
from relbench.tasks import get_task
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd

def entity_features_to_angle(df: pd.DataFrame, n_components: int = 1, angle_range: float = np.pi):
    """
    Convert numeric features of a dataframe to a 1D angle for QRF.
    """
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns
    X = df[numeric_cols].values

    # Normalize to [0, angle_range]
    scaler = MinMaxScaler((0, angle_range))
    X_scaled = scaler.fit_transform(X)

    # Reduce to 1D
    pca = PCA(n_components=n_components)
    X_angle = pca.fit_transform(X_scaled)
    return X_angle.flatten()  # 1D array


def load_relbench(dataset_name: str, task_name: str, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
    """
    Load a RelBench dataset and task, convert entity features to QRF angles.
    
    Returns:
        X_angles: np.ndarray of shape (num_entities,)
        y: np.ndarray of shape (num_entities,)
    """
    print(f"[INFO] Loading dataset '{dataset_name}' task '{task_name}' (split={split})...")
    dataset = get_dataset(dataset_name, download=True)
    task = get_task(dataset_name, task_name, download=True)

    # 1. Access entity table
    # RelBench 1.1.0: entity_table is a string; fetch via dataset DB
    db = dataset.get_db()
    entity_table_name = task.entity_table  # string
    # The database object provides 'tables' as a list; we search for the matching table name
    entity_table = None
    for table in db.tables:
        if table.name == entity_table_name:
            entity_table = table
            break
    if entity_table is None:
        raise ValueError(f"Entity table '{entity_table_name}' not found in database.")

    entity_df = entity_table.to_pandas()

    # 2. Load label table for the split
    label_df = task.get_table(split).to_pandas()

    # 3. Join entity features with labels
    df = entity_df.merge(label_df, on="entity_id", how="inner")

    # 4. Extract angles and labels
    target_col = task.target_column
    X_angles = entity_features_to_angle(df.drop(columns=["entity_id", target_col]))
    y = df[target_col].values

    print(f"[INFO] Loaded {len(y)} samples with 1D angles for QRF.")
    return X_angles, y


def load_relbench_all_splits(dataset_name: str, task_name: str):
    """
    Load train, val, test splits and convert entity features to QRF angles.
    
    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    X_train, y_train = load_relbench(dataset_name, task_name, split="train")
    X_val, y_val = load_relbench(dataset_name, task_name, split="val")
    X_test, y_test = load_relbench(dataset_name, task_name, split="test")
    return X_train, y_train, X_val, y_val, X_test, y_test
