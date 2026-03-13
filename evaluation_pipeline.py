#evaluation_pipeline.py
import os
import numpy as np
import matplotlib.pyplot as plt
from QAttention import quantum_reference_frame_attention
from data_info.datasets.toy_dataset import ToyRelationalQRF
from attention_baselines import evaluate_all_attentions
from main import train_qrf

# ----------------------------
# Unified Evaluation Pipeline
# ----------------------------
def run_pipeline(dataset_names, n_tokens=4, epochs=5, lr=0.05, plot=True, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}

    for ds_name in dataset_names:
        print(f"\n=== Dataset: {ds_name} ===")
        
        # Load dataset using ToyRelationalQRF
        dataset_obj = ToyRelationalQRF(ds_name)

        # Initialise QRF: 2 reference qubits + token qubits (n_tokens)
        qrf = quantum_reference_frame_attention(n_qubits=2 + n_tokens)

        # Train QRF
        theta = train_qrf(qrf, dataset_obj, epochs=epochs, lr=lr)

        # Evaluate on test set
        classical_preds, kernel_preds, qrf_preds, labels = evaluate_all_attentions(dataset_obj, qrf, theta)

        # Compute accuracies
        labels = np.array([dataset_obj.label_encoder.encode(1) if l > 0.5 else dataset_obj.label_encoder.encode(0) for l in labels])
        acc_classical = np.mean(np.array(classical_preds) == labels)
        acc_kernel = np.mean(np.array(kernel_preds) == labels)
        acc_qrf = np.mean(np.array(qrf_preds) == labels)
        print(f"[RESULTS] Classical: {acc_classical:.4f}, Kernel: {acc_kernel:.4f}, QRF: {acc_qrf:.4f}")

        # Save metrics
        all_results[ds_name] = {
            "classical_acc": acc_classical,
            "kernel_acc": acc_kernel,
            "qrf_acc": acc_qrf
        }

        # Optional plot: QRF predictions vs true labels
        if plot:
            plt.figure(figsize=(6,4))
            plt.scatter(range(len(labels)), labels, label="True", alpha=0.7)
            plt.scatter(range(len(qrf_preds)), qrf_preds, label="QRF Predicted", alpha=0.5)
            plt.title(f"{ds_name} - QRF Predictions vs True")
            plt.xlabel("Sample Index")
            plt.ylabel("Label")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"{ds_name}_qrf_plot.png"))
            plt.close()

    print("\n=== Summary Metrics ===")
    for ds, metrics in all_results.items():
        print(f"{ds}: {metrics}")

    return all_results


# ----------------------------
#       Entry Point
# ----------------------------
if __name__ == "__main__":

    # all available datasets:
    # datasets = [
    #     "boston_housing", "california_housing", "citeseer", "cora",
    #     "drug_interactions", "financial_nlp_small", "icml", "nell_sports",
    #     "roofworld20", "toy_cancer", "toy_father", "toy_machines", 
    #     "uwcse", "webkb"
    # ]

    # specifically selected smaller datasets for testing purposes
    # datasets = [
    #     "toy_cancer", "toy_father", "toy_machines"
    # ]

    # smallest of non 'toy' (synthetic) datastes
    datasets = [
        "roofworld20"
    ]

    results = run_pipeline(datasets, n_tokens=4, epochs=3, lr=0.05, plot=True)