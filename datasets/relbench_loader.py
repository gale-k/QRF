# relbench_loader.py

from relbench.datasets import get_dataset
from relbench.tasks import get_task
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd

# def entity_features_to_angle(df: pd.DataFrame, n_components: int = 1, angle_range: float = np.pi):
#     """
#     Convert numeric features of a dataframe to a 1D angle for QRF.
#     """
#     numeric_cols = df.select_dtypes(include=['float', 'int']).columns
#     X = df[numeric_cols].values

#     # Normalize to [0, angle_range]
#     scaler = MinMaxScaler((0, angle_range))
#     X_scaled = scaler.fit_transform(X)

#     # Reduce to 1D
#     pca = PCA(n_components=n_components)
#     X_angle = pca.fit_transform(X_scaled)
#     return X_angle.flatten()  # 1D array

def entity_features_to_angle(
    df: pd.DataFrame,
    n_components: int = 1,
    angle_range: float = np.pi,
    pca=None,
    scaler=None,
    fit=True,
    ):
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    X = df[numeric_cols].values

    if pca is None:
        pca = PCA(n_components=n_components)
    if scaler is None:
        scaler = MinMaxScaler((0, angle_range))

    if fit:
        X_pca = pca.fit_transform(X)
        X_angle = scaler.fit_transform(X_pca)
    else:
        X_pca = pca.transform(X)
        X_angle = scaler.transform(X_pca)

    return X_angle, pca, scaler


# def load_relbench(dataset_name: str, task_name: str, split: str = "train") -> tuple[np.ndarray, np.ndarray]:
def load_relbench(
    dataset_name: str,
    task_name: str,
    split: str = "train",
    pca=None,
    scaler=None,
    fit=True,
    ):

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
    # 1. Get database WITHOUT triggering temporal slicing
    # Disable temporal slicing (RelBench 1.x safe)
    if hasattr(dataset, "test_timestamp"):
        dataset.test_timestamp = None

    db = dataset.get_db()


    entity_table_name = task.entity_table

    entity_table = db.table_dict.get(entity_table_name)
    if entity_table is None:
        raise ValueError(f"Entity table '{entity_table_name}' not found in database.")

    # Convert directly to pandas (no upto(), no query)
    entity_df = entity_table.df.copy()


    # 2. Load label table for the split
    label_df = task.get_table(split).to_pandas()

    # 3. Join entity features with labels
    df = entity_df.merge(label_df, on="entity_id", how="inner")

    # debug check:
    print("Entity rows:", entity_df.shape)
    print("Label rows:", label_df.shape)
    print("Merged rows:", df.shape)


    # 4. Extract angles and labels
    target_col = task.target_column
    # X_angles = entity_features_to_angle(df.drop(columns=["entity_id", target_col]))
    # y = df[target_col].values

    X_angles, pca, scaler = entity_features_to_angle(
    df.drop(columns=["entity_id", target_col]),
    n_components=1,
    pca=pca,
    scaler=scaler,
    fit=fit,
    )
    y = df[target_col].values


    print(f"[INFO] Loaded {len(y)} samples with 1D angles for QRF.")
    # return X_angles, y
    return X_angles, y, pca, scaler


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
