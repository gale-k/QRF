from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
from qiskit.circuit import ParameterVector

# ----------------------------
# QRF Circuit with annotated θ
# ----------------------------
def build_annotated_qrf():
    n_qubits = 4  # 2 reference + 2 token
    n_params = 9
    theta = ParameterVector("θ", n_params)

    qc = QuantumCircuit(n_qubits, 2)

    # Reference qubits (q0, q1)
    qc.h(0)               # Reference frame
    qc.cx(0, 1)

    # Trainable rotations on reference qubits
    qc.ry(theta[0], 0)  # θ0
    qc.rz(theta[4], 0)  # θ4
    qc.ry(theta[1], 1)  # θ1
    qc.rz(theta[5], 1)  # θ5

    # Token qubits (q2, q3) with example angles
    qc.ry(0.5, 2)        # example input
    qc.ry(1.0, 3)        # example input

    # Trainable rotations on token qubits
    qc.ry(theta[2], 2)   # θ2
    qc.rz(theta[6], 2)   # θ6
    qc.ry(theta[3], 3)   # θ3
    qc.rz(theta[7], 3)   # θ7

    # Entanglement with reference
    qc.cx(0, 2)
    qc.cx(1, 3)

    # Relational phase
    qc.cry(theta[8], 2, 3)  # θ8

    # Measurement
    qc.measure([2, 3], [0, 1])

    return qc

# ----------------------------
# Draw and save circuit
# ----------------------------
if __name__ == "__main__":
    qc = build_annotated_qrf()

    # Save as SVG
    circuit_drawer(qc, output='mpl', filename='qrf_annotated.svg')
    # Save as PNG
    circuit_drawer(qc, output='mpl', filename='qrf_annotated.png')

    print("Annotated QRF circuit saved as 'qrf_annotated.svg' and 'qrf_annotated.png'.")