# main.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from datasets.relbench_loader import load_relbench
from QAttention import quantum_reference_frame_attention


# -------------------------------------------------------
# Build relational pairs for QRF attention
# -------------------------------------------------------
def build_relational_pairs(X, y, num_pairs=2000):
    pairs, labels = [], []
    N = len(X)

    for _ in range(num_pairs):
        i, j = np.random.choice(N, 2, replace=False)

        # QRF uses angles → one dimension per entity
        query_angle = float(X[i][0])
        key_angle   = float(X[j][0])

        # Relation label: same class or not
        label = 1 if y[i] == y[j] else 0

        pairs.append([query_angle, key_angle])
        labels.append(label)

    return np.array(pairs), np.array(labels)


# -------------------------------------------------------
# Compute QRF attention scores
# -------------------------------------------------------
def compute_qrf_scores(X_pairs):
    qrf = quantum_reference_frame_attention()
    scores = []

    for q, k in X_pairs:
        circuit = qrf.build_qrf_circuit(q, k)
        score = qrf.attention_score(circuit)
        scores.append(score)

    return np.array(scores).reshape(-1, 1)


# -------------------------------------------------------
# Run QRF on a specific RelBench dataset + task
# -------------------------------------------------------
def run_qrf_experiment(dataset_name, task_name, num_pairs=2000):
    print(f"\n=== Running QRF Attention on {dataset_name} / {task_name} ===")

    # 1. Load RelBench entity table for the task
    X, y = load_relbench(dataset_name, task_name)#, n_features=4)

    # debugging
    print(X)
    print(y)

    # 2. Build QRF relational input pairs
    X_pairs, y_pairs = build_relational_pairs(X, y, num_pairs=num_pairs)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pairs, y_pairs, test_size=0.2, random_state=42
    )

    # 4. Compute QRF relational scores
    X_train_scores = compute_qrf_scores(X_train)
    X_test_scores = compute_qrf_scores(X_test)

    # 5. Train classical classifier on QRF outputs
    clf = LogisticRegression()
    clf.fit(X_train_scores, y_train)

    preds = clf.predict(X_test_scores)
    acc = accuracy_score(y_test, preds)

    print(f"[RESULT] QRF accuracy on relational pair task: {acc:.4f}")
    return acc


# -------------------------------------------------------
# Run across multiple datasets
# -------------------------------------------------------
def main():

    # All datasets available in RelBench are here, with a single corresponding task:
    experiments = [
        ("rel-stack", "user-engagement"),
        ("rel-amazon", "user-churn"),
        ("rel-trial", "study-outcome"),
        ("rel-f1", "driver-position"),
        ("rel-hm", "user-churn"),
        ("rel-event", "user-repeat"),
        ("rel-avito", "user-visits"),
    ]

    results = {}

    for dataset_name, task_name in experiments:
        acc = run_qrf_experiment(dataset_name, task_name)
        results[(dataset_name, task_name)] = acc

    print("\n=== Final QRF Results ===")
    for (dataset, task), acc in results.items():
        print(f"{dataset:<15s} | {task:<20s} | accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
