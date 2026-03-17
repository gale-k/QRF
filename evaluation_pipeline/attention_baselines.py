# attention_baseline.py

import numpy as np
from classical_attention import classical_attention_matrix
from quantum_kernel_attention import quantum_kernel_attention_matrix

from plots import plot_attention_matrix, dataset_plots

def evaluate_all_attentions(dataset, qrf, theta_values=None, eval_samples=100):

    n = len(dataset)
    eval_samples = min(eval_samples, n)

    classical_preds = []
    kernel_preds = []
    qrf_preds = []
    labels = []

    query_angles = []
    key_angles = []

    # Sample subset of dataset
    indices = np.random.choice(n, eval_samples, replace=False)

    for i in indices:
        q, k, label = dataset.get_pair(i)
        query_angles.append(q)
        key_angles.append(k)
        labels.append(label)


    # Classical attention
    classical_matrix = classical_attention_matrix(query_angles, key_angles)

    # get threshold
    c_threshold = np.mean(np.diag(classical_matrix))
    
    # print("[DEBUG] Example classical scores:", classical_matrix[:5,:5])

    for i in range(eval_samples):
        score = classical_matrix[i, i]  # use pairwise attention
        classical_preds.append(1 if score > c_threshold else 0)


    # Quantum kernel attention
    kernel_matrix = quantum_kernel_attention_matrix(query_angles, key_angles, shots=1024) # same amount of shots as qrf

    # get threshold
    k_threshold = np.mean(np.diag(kernel_matrix))

    for i in range(eval_samples):
        score = kernel_matrix[i, i]  # use pairwise attention
        kernel_preds.append(1 if score > k_threshold else 0)

    plot_attention_matrix(classical_matrix,
                      "Classical Attention Matrix",
                      "classical_attention.png")

    plot_attention_matrix(kernel_matrix,
                      "Quantum Kernel Attention Matrix",
                      "kernel_attention.png")

    # QRF attention
    for i in range(eval_samples):

        row_scores = []

        token_angles_i = [query_angles[i], key_angles[i]]

        for j in range(eval_samples):

            token_angles_j = [query_angles[j], key_angles[j]]

            qc = qrf.build_qrf_circuit(token_angles_i + token_angles_j, theta_values)

            score = qrf.attention_score(qc)

            row_scores.append(score)

        pred = np.mean(row_scores)

        qrf_preds.append(1 if pred > 0.5 else 0)

    # Compute accuracies
    labels = np.array(labels)

    classical_acc = np.mean(np.array(classical_preds) == labels)
    kernel_acc = np.mean(np.array(kernel_preds) == labels)
    qrf_acc = np.mean(np.array(qrf_preds) == labels)

    print(f"Classical Attention Accuracy: {classical_acc:.4f}")
    print(f"Quantum Kernel Attention Accuracy: {kernel_acc:.4f}")
    print(f"QRF Attention Accuracy: {qrf_acc:.4f}")

    return classical_preds, kernel_preds, qrf_preds, labels

def evaluate_qrf_with_baselines(dataset_name, dataset, qrf, theta, eval_samples=100):
    """
    Evaluate QRF along with classical and quantum kernel attention on a sampled subset.
    """

    classical_preds, kernel_preds, qrf_preds, labels = evaluate_all_attentions(
        dataset,
        qrf,
        theta_values=theta,
        eval_samples=eval_samples
    )

    # Encode labels to binary for comparison
    labels = np.array([dataset.label_encoder.encode(1) if l > 0.5 else dataset.label_encoder.encode(0) for l in labels])

    acc_classical = np.mean(np.array(classical_preds) == labels)
    acc_kernel = np.mean(np.array(kernel_preds) == labels)
    acc_qrf = np.mean(np.array(qrf_preds) == labels)

    # print(f"[RESULTS] Classical: {acc_classical:.4f}, Kernel: {acc_kernel:.4f}, QRF: {acc_qrf:.4f}")

    dataset_plots(labels, qrf_preds, dataset_name)
    
    return acc_classical, acc_kernel, acc_qrf
