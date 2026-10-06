import numpy as np


def householder_vector(x):
    """
    x: 1D array of floats

    Returns:
        A 1D array v of the same length such that the reflection across
        the plane perpendicular to v sends x to
        [-sign(x[0]) * norm(x), 0, ..., 0], with sign(0) = 1. Returns the
        zero vector when x is the zero vector. x must not be modified.
    """
    # TODO: Implement the direction from Theory, with the sign chosen to
    # avoid cancellation in the first entry.
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x)
    v = x.copy()
    if norm == 0.0:
        return v
    sign = 1.0 if x[0] >= 0 else -1.0
    v[0] += sign * norm
    return v


def householder_matrix(v):
    """
    v: 1D array of floats, the mirror direction

    Returns:
        The n x n matrix I - 2 v v^T / (v^T v). Returns the identity when
        v is the zero vector.
    """
    # TODO: Implement the formula from Theory.
    v = np.asarray(v, dtype=float)
    n = len(v)
    denominator = v @ v
    if denominator == 0.0:
        return np.eye(n)
    return np.eye(n) - 2.0 * np.outer(v, v) / denominator


def apply_householder(v, a):
    """
    v: 1D array of floats, the mirror direction
    a: 1D vector or 2D matrix with len(v) rows

    Returns:
        The reflection of a (each column, for a matrix) with the same
        shape as a, computed without building the n x n matrix. Returns a
        copy of a when v is the zero vector. v and a must not be modified.
    """
    # TODO: Implement a - 2 v (v^T a) / (v^T v) from Theory.
    v = np.asarray(v, dtype=float)
    a = np.asarray(a, dtype=float)
    denominator = v @ v
    if denominator == 0.0:
        return a.copy()
    projection = v @ a
    if a.ndim == 1:
        return a - 2.0 * projection * v / denominator
    return a - 2.0 * np.outer(v, projection) / denominator
