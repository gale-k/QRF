# QRFAttention.py
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from qiskit_aer import Aer

class quantum_reference_frame_attention:
    def __init__(self, n_qubits=4, n_params=9):
        self.n_qubits = n_qubits
        self.theta = ParameterVector("θ", n_params)
    
    def get_circuit(self, query_angle, key_angle):
        qc = QuantumCircuit(self.n_qubits)
        # reference frame (qubits 0 & 1)
        qc.h(0)     # put qubit 1 into superposition
        qc.cx(0, 1) # entangle qubits 0 & 1
        # established shared reference frame (like a 'quantum coordinate system')

        # encode query/key (qubits 2 & 3) into quantum states
        qc.ry(query_angle, 2)
        qc.ry(key_angle, 3)

        # parameterised layers
        for i in range(4):
            qc.ry(self.theta[i], i)     # trainable y rotation on each qubit (magnitude of attention)
            qc.rz(self.theta[i+4], i)   # trainable z rotation on each qubit (phase)
        # learnable paramters allow model to adapt 

        # establish relational entanglement
        qc.cx(0, 2)     # entangle reference (qubit 0) with query (qubit 2)
        qc.cx(1, 3)     # entangle reference (qubit 1) with key (qubit 3)
        # created quantum correlations between reference frame and query & key
        # ensures query & key are compared relative to same reference

        # encode relational phase
        qc.cry(self.theta[8], 2, 3)     # controlled y rotation betwen query & key

        return qc
    
    def get_qrf_attention(self, qc, shots=1024):
        total = 0
        runs = 3
        for _ in range(runs):
            simulator = Aer.get_backend('qasm_simulator')
            result = simulator.run(qc, shots=shots).result()
            counts = result.get_counts()
            # probability that tokens are in the same state
            total += (counts.get('00', 0) + counts.get('11', 0)) / shots
        return total/runs
    
    def get_feature_map(self, num_features, query, key):
        # convert to a feature map that takes input angles
        feature_map = QuantumCircuit(num_features)
        feature_map.compose(self.get_circuit(query, key), inplace=True)
        return feature_map
    
    def get_measurement_circuit(self, query_angle, key_angle):
        qc = self.get_circuit(query_angle, key_angle)
        # measure in different bases for richer information
        # measure token qubits (2-3) - query & key
        qc.h([2,3])  # Hadamard basis measurement
        qc.measure([2,3], [0,1])
        return qc
