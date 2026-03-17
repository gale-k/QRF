# classical_pipeline.py

import numpy as np

def angle_to_vector(theta):
    # Embed a scalar angle as 2D vector for classical attention
    return np.array([np.cos(theta), np.sin(theta)])

def classical_attention_matrix(query_angles, key_angles):
    n = len(query_angles)
    A = np.zeros((n, n))
    
    # convert angles to vectors
    queries = np.array([angle_to_vector(q) for q in query_angles])
    keys = np.array([angle_to_vector(k) for k in key_angles])
    
    # compute raw attention scores
    for i in range(n):
        for j in range(n):
            score = np.dot(queries[i], keys[j])
            A[i, j] = np.exp(score)
    
    # normalise per row
    A /= A.sum(axis=1, keepdims=True)
    return A