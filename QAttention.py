# QAttention.py
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from qiskit_aer import Aer

N_PARAMS = 9

class quantum_reference_frame_attention:
    def __init__(self, n_qubits=4, n_params=N_PARAMS):
        """
        Qubits:
        0-1 : reference frame (entangled)
        2   : query
        3   : key
        """
        self.n_qubits = n_qubits
        self.theta = ParameterVector("θ", n_params)
    
    def prepare_reference_frame(self, qc):
        qc.h(0)
        qc.cx(0, 1)
        return qc

    def encode_tokens(self, qc, query_angle, key_angle):
        qc.ry(query_angle, 2)
        qc.ry(key_angle, 3)
        return qc
    
    def add_trainable_layers(self, qc):
        # Add 9 trainable parameters on each qubit
        for i in range(4):
            qc.ry(self.theta[i], i)
            qc.rz(self.theta[i+4], i)
        return qc
    
    def entangle_reference_with_tokens(self, qc):
        qc.cx(0, 2)
        qc.cx(1, 3)
        return qc

    def add_relational_phase(self, qc):
        qc.cry(self.theta[8], 2, 3)
        return qc

    # def build_qrf_circuit(self, query_angle, key_angle):
    #     qc = QuantumCircuit(self.n_qubits, 2)

    #     self.prepare_reference_frame(qc)
    #     self.encode_tokens(qc, query_angle, key_angle)
    #     self.add_trainable_layers(qc)
    #     self.entangle_reference_with_tokens(qc)
    #     self.add_relational_phase(qc)

    #     # measurement on token qubits only
    #     qc.measure([2, 3], [0, 1])
    #     return qc
    
    def build_qrf_circuit(self, query_angle, key_angle):
        qc = QuantumCircuit(self.n_qubits, 2)

        self.prepare_reference_frame(qc)
        self.encode_tokens(qc, query_angle, key_angle)
        self.add_trainable_layers(qc)
        self.entangle_reference_with_tokens(qc)
        self.add_relational_phase(qc)

        # measurement on token qubits only
        qc.measure([2, 3], [0, 1])

        # bind all 9 trainable parameters to numeric values
        param_dict = {self.theta[i]: 0.1 for i in range(len(self.theta))}

        qc = qc.assign_parameters(param_dict)

        return qc


    def attention_score(self, qc, shots=2048):
        backend = Aer.get_backend("aer_simulator")
        result = backend.run(qc, shots=shots).result()
        counts = result.get_counts()

        # token (2,3) outcomes are in last 2 bits
        score = 0
        for bitstring, count in counts.items():
            token_bits = bitstring[-2:]  # last 2 bits
            if token_bits in ["00", "11"]:
                score += count

        return score / shots

