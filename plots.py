# plots.py

import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use("seaborn-v0_8-paper")

def dataset_plots(labels, qrf_preds, dataset_name):
    plt.figure(figsize=(6,4))
    plt.scatter(range(len(labels)), labels, label="True", alpha=0.7)
    plt.scatter(range(len(qrf_preds)), qrf_preds, label="QRF Predicted", alpha=0.5)
    plt.title(f"{dataset_name} - QRF Predictions vs True")
    plt.xlabel("Sample Index")
    plt.ylabel("Label")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results/qrf_plots", f"{dataset_name}_qrf_plot.png"))
    plt.close()

def plot_accuracy_comparison(dataset_name, classical_acc, kernel_acc, qrf_acc):

    models = ["Classical Attention", "Quantum Kernel", "QRF Attention"]
    accuracies = [classical_acc, kernel_acc, qrf_acc]

    plt.figure(figsize=(6,4))

    plt.bar(models, accuracies)

    plt.ylabel("Accuracy")
    plt.title(f"{dataset_name} - Attention Mechanism Performance Comparison")

    plt.ylim(0,1)

    plt.tight_layout()
    plt.savefig(os.path.join("results", "attention_accuracy", f"{dataset_name}_attention_accuracy_comparison.png"), dpi=300)

    plt.close()


import matplotlib.pyplot as plt
import numpy as np

def plot_dataset_comparison(all_results):

    datasets = list(all_results.keys())

    classical = [all_results[d]["classical_acc"] for d in datasets]
    kernel = [all_results[d]["kernel_acc"] for d in datasets]
    qrf = [all_results[d]["qrf_acc"] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.25

    plt.figure()

    plt.bar(x - width, classical, width, label="Classical Attention")
    plt.bar(x, kernel, width, label="Quantum Kernel")
    plt.bar(x + width, qrf, width, label="QRF Attention")

    plt.ylabel("Accuracy")
    plt.title("Attention Mechanism Performance Across Datasets")

    plt.xticks(x, datasets, rotation=30)
    plt.ylim(0, 1)

    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join("results","dataset_comparison", "dataset_accuracy_comparison.png"), dpi=300)

    plt.close()


def plot_attention_matrix(matrix, title, filename):

    plt.figure()

    plt.imshow(matrix)
    plt.colorbar()

    plt.title(title)
    plt.xlabel("Keys")
    plt.ylabel("Queries")

    plt.tight_layout()
    plt.savefig(os.path.join("results","attention_matrix", filename), dpi=300)

    plt.close()

import matplotlib.pyplot as plt

def plot_qrf_training_loss(loss_history, filename):

    epochs = range(1, len(loss_history)+1)

    plt.figure()

    plt.plot(epochs, loss_history, marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("QRF Training Loss vs Epoch")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join("results", "epoch_loss", f"{filename}_qrf_training_loss.png"), dpi=300)

    plt.close()
