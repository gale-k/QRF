# wikidbs_loader.py

from wikidbs import WikiDB
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def load_wikidbs(name, n_features=4):
    """
    Loads a real-world relational dataset from WikiDBs.
    - name: e.g., 'airport', 'book', 'film'
    """

    db = WikiDB(name)        # loads SQLite DB
    tables = db.tables()

    # Choose primary table = one with most rows
    main_table = max(tables, key=lambda t: db.shape(t)[0])
    df = db.read(main_table)

    # find a target column automatically
    target = None
    for col in df.columns:
        if "category" in col.lower() or "label" in col.lower() or "class" in col.lower():
            target = col
            break

    if target is None:
        # fallback: use unsupervised "cluster label"
        target = df.columns[-1]

    X = df.drop(columns=[target]).select_dtypes("number").fillna(0).values
    y = df[target].values

    X = PCA(n_components=n_features).fit_transform(X)
    X = MinMaxScaler((0, 1)).fit_transform(X)

    return X, y
