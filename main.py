from QAttention import quantum_reference_frame_attention
from data_info.relational_loader import load_relational_dataset


def main():

    # singular dataset for testing purposes
    dataset_name = "toy_machines" 
    dataset = load_relational_dataset(dataset_name)

    qrf = quantum_reference_frame_attention()

    print(f"\n[INFO] Running QRF on {dataset_name}...\n")

    for i in range(min(len(dataset), 30)):

        query_angle, key_angle, label = dataset.get_pair(i)

        qc = qrf.build_qrf_circuit(
            query_angle=query_angle,
            key_angle=key_angle
        )

        attention = qrf.attention_score(qc)

        print(f"[{i}] Attention: {attention:.4f} | Label: {label:.4f}")


if __name__ == "__main__":
    main()
