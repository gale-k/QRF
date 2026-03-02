from git.QRF.relational_dataset import RelationalDataset
from QAttention import quantum_reference_frame_attention

dataset = RelationalDataset(n_samples=50, seq_len=4)
qrf = quantum_reference_frame_attention(n_qubits=6)  # 2 reference + 4 token qubits

# single sample
query, key, label = dataset.get_pair(0)
qc = qrf.build_qrf_circuit(query, key)
score = qrf.attention_score(qc)
print("QRF attention score:", score, "Label:", label)