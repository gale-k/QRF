from datasets.dataset_api import load_relational_dataset
from QAttention import QuantumReferenceFrameAttention
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def build_rel_pairs(X, y, num_pairs=1000):
    N = len(X)
    pairs, labels = [], []
    for _ in range(num_pairs):
        i, j = np.random.choice(N, 2, replace=False)
        pairs.append([X[i][0], X[j][0]])   # use angle from PCA
        labels.append(int(y[i] == y[j]))
    return np.array(pairs), np.array(labels)

def compute_qrf_scores(pairs):
    qrf = QuantumReferenceFrameAttention()
    scores = []
    for q, k in pairs:
        qc = qrf.build_qrf_circuit(float(q), float(k))
        scores.append(qrf.attention_score(qc))
    return np.array(scores).reshape(-1, 1)

def run(dataset_source, name):
    print(f"\n=== Running QRF on {dataset_source}:{name} ===")

    X, y = load_relational_dataset(dataset_source, name, n_features=4)

    pairs, labels = build_rel_pairs(X, y)
    X_train, X_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.2)

    train_scores = compute_qrf_scores(X_train)
    test_scores = compute_qrf_scores(X_test)

    clf = LogisticRegression().fit(train_scores, y_train)
    preds = clf.predict(test_scores)
    print("Accuracy:", accuracy_score(y_test, preds))

def main():
    run("relbench", "rel-f1")
    run("relbench", "rel-airbnb")
    run("rdbench", "credit")
    run("wikidbs", "airport")

if __name__ == "__main__":
    main()
