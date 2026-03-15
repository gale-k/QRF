from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.circuit import ParameterVector
import numpy as np

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
        self.n_params = n_params
        self.theta = ParameterVector("θ", n_params)

    
    def prepare_reference_frame(self, qc):
        qc.h(0)
        qc.cx(0, 1)
        return qc

    # def encode_tokens(self, qc, query_angle, key_angle):
    #     qc.ry(query_angle, 2)
    #     qc.ry(key_angle, 3)
    #     return qc

    def encode_tokens(self, qc, token_angles, start_qubit=2):
        """
        token_angles: list or array of angles for each token
        start_qubit: which qubit index to start encoding
        """
        for i, angle in enumerate(token_angles):
            qc.ry(angle, start_qubit + i)
        return qc
    
    def add_trainable_layers(self, qc):
        # add 9 trainable parameters on each qubit
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
    
    # def build_qrf_circuit(self, query_angle, key_angle, theta_values=None):
    #     """
    #     theta_values: optional list of numeric parameters to bind
    #     """
    #     qc = QuantumCircuit(self.n_qubits, 2)

    #     # Build the QRF circuit
    #     self.prepare_reference_frame(qc)
    #     self.encode_tokens(qc, query_angle, key_angle)
    #     self.add_trainable_layers(qc)
    #     self.entangle_reference_with_tokens(qc)
    #     self.add_relational_phase(qc)

    #     # Measure token qubits only
    #     qc.measure([2, 3], [0, 1])

    #     # Bind trainable parameters if numeric values are provided
    #     if theta_values is not None:
    #         param_dict = {self.theta[i]: theta_values[i] for i in range(len(self.theta))}
    #         qc = qc.assign_parameters(param_dict)

    #     return qc

    def build_qrf_circuit(self, token_angles, theta_values=None):
        """
        token_angles: list of angles for each token (multi-token sequences)
        n_qubits = len(token_angles) + number of reference qubits
        """

        n_qubits_needed = 2 + len(token_angles)  # 2 reference qubits
        if n_qubits_needed > self.n_qubits:
            raise ValueError(
                f"QRF initialized with {self.n_qubits} qubits, "
                f"but need {n_qubits_needed} for this token sequence."
            )

        qc = QuantumCircuit(self.n_qubits, len(token_angles))

        # --- Reference frame ---
        self.prepare_reference_frame(qc)

        # --- Encode tokens ---
        self.encode_tokens(qc, token_angles, start_qubit=2)

        # --- Add trainable layers ---
        self.add_trainable_layers(qc)

        # --- Entangle reference with tokens ---
        self.entangle_reference_with_tokens(qc)

        # --- Add relational phase ---
        self.add_relational_phase(qc)

        # --- Measure token qubits only ---
        qc.measure(range(2, 2 + len(token_angles)), range(len(token_angles)))

        # --- Bind trainable parameters if provided ---
        if theta_values is not None:
            if len(theta_values) != len(self.theta):
                raise ValueError(
                    f"theta_values length ({len(theta_values)}) "
                    f"must match number of circuit parameters ({len(self.theta)})"
                )
            param_dict = {self.theta[i]: theta_values[i] for i in range(len(self.theta))}
            qc = qc.assign_parameters(param_dict)

        return qc

    def attention_score(self, qc, shots=1024):
        backend = Aer.get_backend("aer_simulator")
        result = backend.run(qc, shots=shots).result()
        counts = result.get_counts()

        # Consider "00" and "11" as aligned states
        score = 0
        for bitstring, count in counts.items():
            token_bits = bitstring[-2:]  # last 2 bits are token qubits
            if token_bits in ["00", "11"]:
                score += count

        return score / shots

    def compute_attention_matrix(self, query_angles, key_angles, theta_values=None):
        """
        Compute full n x n attention matrix for sequences
        query_angles, key_angles: lists of angles for each token
        """
        n = len(query_angles)
        A = np.zeros((n, n))

        # for i in range(n):
        #     for j in range(n):
        #         qc = self.build_qrf_circuit(query_angles[i], key_angles[j], theta_values)
        #         A[i, j] = self.attention_score(qc)
        # return A
    
        max_samples = 50

        indices = np.random.choice(n, max_samples, replace=False)

        for i in indices:
            for j in indices:
                qc = self.build_qrf_circuit(query_angles[i], key_angles[j], theta_values)
                A[i, j] = self.attention_score(qc)
        return A


    @staticmethod
    def binary_cross_entropy(pred, target, eps=1e-8):
        pred = np.clip(pred, eps, 1 - eps)
        return -(target * np.log(pred) + (1 - target) * np.log(1 - pred))