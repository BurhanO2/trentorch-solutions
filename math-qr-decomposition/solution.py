import numpy as np


def qr_decompose(A):
    """
    A: an (m, n) NumPy array with linearly independent columns

    Returns:
        (Q, R): Q is (m, n) with orthonormal columns, R is (n, n)
        upper-triangular, such that Q @ R reconstructs A exactly.
    """
    # TODO: Implement Gram-Schmidt column by column, recording each
    # projection coefficient into R and each leftover norm into R's
    # diagonal, from Theory.
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].astype(float).copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R
