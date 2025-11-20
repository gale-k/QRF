# relbench_loader.py

from relbench.data import load_dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def load_relbench(name, n_features=4):
    """
    Load a RelBench dataset and return (X, y)
    Works for:
        - rel-f1
        - rel-airbnb
        - rel-rossmann
        - rel-walmart
    """

    ds = load_dataset(name)   # returns a relational DB object

    # Usually the "primary entity" is ds.main_table
    main = ds.get_table(ds.main_table)

    # Features & labels from the target task
    X = main.features.drop(columns=[main.target]).values
    y = main.features[main.target].values

    # Dimensionality reduction for quantum embedding
    X = PCA(n_components=n_features).fit_transform(X)
    X = MinMaxScaler((0, 1)).fit_transform(X)

    return X, y
