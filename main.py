from QAttention import quantum_reference_frame_attention
from data_info.relational_loader import load_relational_dataset
from attention_baselines import evaluate_all_attentions

import numpy as np

def main():
    # Load dataset
    dataset_name = "boston_housing"
    dataset = load_relational_dataset(dataset_name)

    # For sequences with 2 query/key pairs
    n_tokens = 4  # query_i, key_i, query_j, key_j
    # Initialize QRF attention
    qrf = quantum_reference_frame_attention(n_qubits=2 + n_tokens)  # 2 reference + 4 tokens

    print("\nTraining QRF...\n")
    theta = train_qrf(qrf, dataset, epochs=5, lr=0.05)

    print("\nEvaluating all attention mechanisms...\n")
    classical_preds, kernel_preds, qrf_preds, labels = evaluate_all_attentions(dataset, qrf, theta)

# ----------------------------
# QRF Training
# ----------------------------
def train_qrf(qrf, dataset, epochs=10, lr=0.1):
    # Randomly initialize 9 trainable theta parameters
    theta = np.random.uniform(0, 2*np.pi, len(qrf.theta))
    print(f"Initial theta length: {len(theta)}")

    for epoch in range(epochs):
        total_loss = 0

        for i in range(len(dataset)):
            # Extract a single pair
            query_angle, key_angle, label = dataset.get_pair(i)
            token_angles = [query_angle, key_angle]  # Pack angles as a list

            # Build QRF circuit and compute attention
            qc = qrf.build_qrf_circuit(token_angles, theta)
            pred = qrf.attention_score(qc)

            # Compute loss
            loss = binary_cross_entropy(pred, label)
            total_loss += loss

            # SPSA gradient approximation
            delta = np.random.choice([-1, 1], size=len(theta))
            epsilon = 0.1

            theta_plus = theta + epsilon * delta
            theta_minus = theta - epsilon * delta

            qc_plus = qrf.build_qrf_circuit(token_angles, theta_plus)
            qc_minus = qrf.build_qrf_circuit(token_angles, theta_minus)

            loss_plus = binary_cross_entropy(qrf.attention_score(qc_plus), label)
            loss_minus = binary_cross_entropy(qrf.attention_score(qc_minus), label)

            grad_estimate = (loss_plus - loss_minus) / (2 * epsilon) * delta
            theta -= lr * grad_estimate

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | theta: {theta}")

    return theta

# ----------------------------
# Utility
# ----------------------------
def binary_cross_entropy(pred, target):
    eps = 1e-9
    pred = np.clip(pred, eps, 1 - eps)
    return -(target * np.log(pred) + (1 - target) * np.log(1 - pred))

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    main()