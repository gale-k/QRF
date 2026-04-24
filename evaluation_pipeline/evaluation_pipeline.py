# evaluation_pipeline.py

from attention_models.quantum_reference_frame_attention import quantum_reference_frame_attention
from data_info.datasets.toy_dataset import RelationalQRF
from evaluation_pipeline.attention_baselines import evaluate_qrf_with_baselines
from evaluation_pipeline.train_qrf_batched import train_qrf_batched

from plots import plot_accuracy_comparison, plot_dataset_comparison, plot_qrf_training_loss


# main pipeline
def run_pipeline(
        datasets,
        epochs=5,
        batch_size=16,
        key_samples=4,
        eval_samples=100,
        mode="full"
    ):

    all_results = {}

    for dataset_name in datasets:
        print(f"\n=== Dataset: {dataset_name} ===")

        # load relational dataset wrapper
        dataset_obj = RelationalQRF(dataset_name)

        # determine number of qubits needed for QRF
        # each token pair has query + key, and we concatenate two pairs in attention
        n_tokens_per_pair = 2  # query + key
        total_tokens = 2 * n_tokens_per_pair  # i + j concatenation in QRF evaluation
        n_qubits = 2 + total_tokens  # 2 reference qubits + token qubits

        # initialise QRF attention model with enough qubits
        qrf = quantum_reference_frame_attention(n_qubits=n_qubits, mode=mode)

        # train QRF using mini-batches
        theta, loss = train_qrf_batched(
            qrf,
            dataset_obj,
            epochs=epochs,
            batch_size=batch_size,
            key_samples=key_samples
        )

        plot_qrf_training_loss(loss, dataset_name)

        # evaluate QRF + baselines
        acc_classical, acc_kernel, acc_qrf = evaluate_qrf_with_baselines(
            dataset_name,
            dataset_obj,
            qrf,
            theta,
            eval_samples=eval_samples
        )

        plot_accuracy_comparison(dataset_name, acc_classical, acc_kernel, acc_qrf)

        all_results[dataset_name] = {
            "classical_acc": acc_classical,
            "kernel_acc": acc_kernel,
            "qrf_acc": acc_qrf
        }
        

    print("\n=== Summary Metrics ===")
    for ds, metrics in all_results.items():
        print(f"{ds}: {metrics}")


    return all_results

# entry point
def main_pipeline():

    modes = ["full", "no_entanglement", "no_reference"]

    all_ablation_results = {}

    datasets = [
        "toy_cancer", "toy_father", "toy_machines"
    ]

    # datasets = [
    #     "boston_housing", "california_housing", "citeseer", "cora",
    #     "drug_interactions", "financial_nlp_small", "icml", "nell_sports",
    #     "roofworld20", "uwcse", "webkb"
    # ]

    # results = run_pipeline(
    #     datasets=datasets,
    #     epochs=20,
    #     batch_size=16,
    #     key_samples=6,
    #     eval_samples=100,
    #     mode=mode
    # )

    for mode in modes:
        print(f"\n\n===== RUNNING MODE: {mode.upper()} =====")

        results = run_pipeline(
            datasets=datasets,
            epochs=20,
            batch_size=16,
            key_samples=6,
            eval_samples=100,
            mode=mode
        )


