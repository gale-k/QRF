from qiskit import QuantumCircuit
from qiskit_aer import Aer
import numpy as np

# def quantum_kernel_score(theta_i, theta_j, shots=1024):
#     """Compute a measurement-based approximation of |<psi_i|psi_j>|^2"""
#     backend = Aer.get_backend("aer_simulator")

#     # prepare first state
#     qc_i = QuantumCircuit(1, 1)
#     qc_i.ry(theta_i, 0)
#     qc_i.measure(0, 0)
#     result_i = backend.run(qc_i, shots=shots).result()
#     counts_i = result_i.get_counts()
#     # probability of |0> for first state
#     p0_i = counts_i.get("0", 0) / shots

#     # prepare second state
#     qc_j = QuantumCircuit(1, 1)
#     qc_j.ry(theta_j, 0)
#     qc_j.measure(0, 0)
#     result_j = backend.run(qc_j, shots=shots).result()
#     counts_j = result_j.get_counts()
#     p0_j = counts_j.get("0", 0) / shots

#     # approximate fidelity as probability overlap
#     score = np.sqrt(p0_i * p0_j) + np.sqrt((1 - p0_i) * (1 - p0_j))
#     return score**2

def quantum_kernel_score(theta_i, theta_j, shots=1024):

    backend = Aer.get_backend("aer_simulator")

    qc = QuantumCircuit(1,1)

    qc.ry(theta_i,0)
    qc.ry(-theta_j,0)

    qc.measure(0,0)

    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()

    return counts.get("0",0) / shots

def quantum_kernel_attention_matrix(query_angles, key_angles, shots=1024):

    n = len(query_angles)
    A = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            A[i,j] = quantum_kernel_score(query_angles[i], key_angles[j], shots)

    A /= A.sum(axis=1, keepdims=True)

    return A