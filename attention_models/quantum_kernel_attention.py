# quantum_kernel_attention.py

from qiskit import QuantumCircuit
from qiskit_aer import Aer
import numpy as np

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
