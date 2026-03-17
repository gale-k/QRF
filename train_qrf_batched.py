# train_qrf_batched.py

import numpy as np
from batch_utils import sample_batch, sample_keys
from parameter_shift import parameter_shift_gradient


def train_qrf_batched(qrf, dataset, epochs=10, batch_size=16, key_samples=4, lr=0.05):

    theta = np.random.uniform(0, 2*np.pi, qrf.n_params)

    print("Initial theta length:", len(theta))
    loss_history = []

    for epoch in range(epochs):

        batch = sample_batch(dataset, batch_size)

        total_loss = 0
        
        for query_angle, _, label in batch:

            keys = sample_keys(dataset, key_samples)

            scores = []
            grads = []

            for key_angle in keys:

                qc = qrf.build_qrf_circuit([query_angle, key_angle], theta)
                score = qrf.attention_score(qc)
                scores.append(score)

                grad = parameter_shift_gradient(qrf, query_angle, key_angle, theta)
                grads.append(grad)

            pred = np.mean(scores)

            loss = (pred - label) ** 2
            total_loss += loss

            grad_mean = np.mean(grads, axis=0)

            theta -= lr * grad_mean

        avg_loss = total_loss / batch_size
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    return theta, loss_history