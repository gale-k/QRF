import numpy as np
from QAttention import quantum_reference_frame_attention
from data_info.relational_loader import load_relational_dataset
from classical_attention import classical_attention_matrix
from quantum_kernel_attention import quantum_kernel_attention_matrix

# ----------------------------
# Unified Evaluation (Updated for multi-token sequences)
# ----------------------------
def evaluate_all_attentions(dataset, qrf, theta_values=None):
    n = len(dataset)
    
    classical_preds = []
    kernel_preds = []
    qrf_preds = []
    labels = []

    query_angles = []
    key_angles = []

    # Extract all pairs
    for i in range(n):
        q, k, label = dataset.get_pair(i)
        query_angles.append(q)
        key_angles.append(k)
        labels.append(label)

    # ----------------------------
    # Classical attention
    # ----------------------------
    from classical_attention import classical_attention_matrix
    classical_matrix = classical_attention_matrix(query_angles, key_angles)
    for i in range(n):
        pred = np.mean(classical_matrix[i])
        classical_preds.append(1 if pred > 0.5 else 0)

    # ----------------------------
    # Quantum kernel attention
    # ----------------------------
    from quantum_kernel_attention import quantum_kernel_attention_matrix
    kernel_matrix = quantum_kernel_attention_matrix(query_angles, key_angles, shots=1024)
    for i in range(n):
        pred = np.mean(kernel_matrix[i])
        kernel_preds.append(1 if pred > 0.5 else 0)

    # ----------------------------
    # QRF attention (multi-token)
    # ----------------------------
    for i in range(n):
        row_scores = []
        # Pack query and key angles as a multi-token input
        token_angles_i = [query_angles[i], key_angles[i]]
        for j in range(n):
            token_angles_j = [query_angles[j], key_angles[j]]
            qc = qrf.build_qrf_circuit(token_angles_i + token_angles_j, theta_values)
            score = qrf.attention_score(qc)
            row_scores.append(score)
        pred = np.mean(row_scores)
        qrf_preds.append(1 if pred > 0.5 else 0)

    # ----------------------------
    # Compute accuracies
    # ----------------------------
    labels = np.array(labels)
    classical_acc = np.mean(np.array(classical_preds) == labels)
    kernel_acc = np.mean(np.array(kernel_preds) == labels)
    qrf_acc = np.mean(np.array(qrf_preds) == labels)

    print(f"Classical Attention Accuracy: {classical_acc:.4f}")
    print(f"Quantum Kernel Attention Accuracy: {kernel_acc:.4f}")
    print(f"QRF Attention Accuracy: {qrf_acc:.4f}")
    
    return classical_preds, kernel_preds, qrf_preds, labels