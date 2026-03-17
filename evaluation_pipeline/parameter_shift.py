import numpy as np


def parameter_shift_gradient(qrf, query_angle, key_angle, theta):

    gradients = np.zeros(len(theta))

    shift = np.pi / 2

    for i in range(len(theta)):

        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[i] += shift
        theta_minus[i] -= shift

        qc_plus = qrf.build_qrf_circuit([query_angle, key_angle], theta_plus)
        qc_minus = qrf.build_qrf_circuit([query_angle, key_angle], theta_minus)

        f_plus = qrf.attention_score(qc_plus)
        f_minus = qrf.attention_score(qc_minus)

        gradients[i] = (f_plus - f_minus) / 2

    return gradients
