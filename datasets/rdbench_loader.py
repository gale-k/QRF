# rdbench_loader.py

from rdbench.data import RDBench
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def load_rdbench(name, n_features=4):
    """
    Load one of the RDBench relational databases.
    Example names:
       - credit
       - loan
       - airbnb (RDBench version)
    """

    db = RDBench(name)
    table = db.main_table  # primary entity table

    df = db.load_table(table)
    target = db.target_column(table)

    X = df.drop(columns=[target]).values
    y = df[target].values

    X = PCA(n_components=n_features).fit_transform(X)
    X = MinMaxScaler((0, 1)).fit_transform(X)

    return X, y
