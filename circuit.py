# produce a circuit image file from your existing Qiskit circuit
from QAttention import QuantumCircuit
# construct the circuit (example: follow your QAttention.py)
qc = QuantumCircuit(4,4)
# 0-1 reference frame
qc.h(0)
qc.cx(0,1)
# encode query/key on qubits 2,3
qc.ry(0.8,2)   # use symbolic or numeric angles for display
qc.ry(1.2,3)
# param layers (example placeholders)
qc.ry(0.1,0); qc.rz(0.2,0)
qc.ry(0.3,1); qc.rz(0.4,1)
qc.ry(0.5,2); qc.rz(0.6,2)
qc.ry(0.7,3); qc.rz(0.8,3)
# relational entanglement
qc.cx(0,2)
qc.cx(1,3)
# controlled rotation (CRY) - Qiskit supports crx/crz; for cry use qiskit.circuit.library
from qiskit.circuit.library import CRYGate
qc.append(CRYGate(0.9), [2,3])
# measurement (optional)
qc.measure([0,1,2,3],[0,1,2,3])

# Draw and save
qc.draw(output='mpl', filename='qrf_circuit.png', scale=1.4)
# or for svg
qc.draw(output='mpl').savefig('qrf_circuit.svg')
