# circuit.py

from attention_models.quantum_reference_frame_attention import quantum_reference_frame_attention
import matplotlib.pyplot as plt
import numpy as np
import os

# Example token angles for 2 tokens (query + key)
token_angles = [0.5, 1.0]  # radians

# Initialize QRF with 2 reference qubits + 2 token qubits
qrf = quantum_reference_frame_attention(n_qubits=4)

# Build the QRF circuit with random theta values
theta_values = np.random.uniform(0, 2 * np.pi, len(qrf.theta))
qc = qrf.build_qrf_circuit(token_angles, theta_values)


# save as SVG
qc.draw(output='mpl')  # matplotlib figure
plt.savefig(os.path.join("circuits","qrf_circuit.svg"), bbox_inches='tight')
print("[INFO] Saved QRF circuit as qrf_circuit.svg")


# save as PNG
plt.savefig(os.path.join("circuits", "qrf_circuit.png"), bbox_inches='tight', dpi=300)
print("[INFO] Saved QRF circuit as qrf_circuit.png")
