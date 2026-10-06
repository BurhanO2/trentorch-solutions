import numpy as np

from _load import load_solution

householder_vector = load_solution("math-householder-reflections").householder_vector
apply_householder = load_solution("math-householder-reflections").apply_householder


def householder_qr(a):
    """
    a: m x n float array with m >= n

    Returns:
        (Q, R): Q is m x m and orthogonal, R is m x n with exact zeros
        below the diagonal, and Q @ R equals a. Uses one Householder
        mirror per column; skips a column whose mirror direction is zero.
        a must not be modified.
    """
    # TODO: Clear each column below the diagonal with a mirror, as in Theory,
    # and accumulate the mirrors into Q.
    r = np.array(a, dtype=float)
    m, n = r.shape
    q = np.eye(m)
    for k in range(min(m - 1, n)):
        v = householder_vector(r[k:, k])
        if not np.any(v):
            continue
        r[k:, :] = apply_householder(v, r[k:, :])
        q[:, k:] = apply_householder(v, q[:, k:].T).T
    return q, np.triu(r)


def back_substitution(r, y):
    """
    r: n x n upper-triangular float array with a nonzero diagonal
    y: 1D float array of length n

    Returns:
        The 1D array x with r @ x == y, found from the last row upward
        with a loop (do not call np.linalg.solve).
    """
    # TODO: Solve from the last row upward, as in Theory.
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - r[i, i + 1 :] @ x[i + 1 :]) / r[i, i]
    return x


def least_squares_qr(a, b):
    """
    a: m x n float array with full column rank, m >= n
    b: 1D float array of length m

    Returns:
        The 1D array x of length n minimising the squared error of
        a @ x - b, using householder_qr and back_substitution. Do not form
        a.T @ a.
    """
    # TODO: Reduce to the triangular system from Theory.
    a = np.asarray(a, dtype=float)
    n = a.shape[1]
    q, r = householder_qr(a)
    c = q.T @ np.asarray(b, dtype=float)
    return back_substitution(r[:n, :n], c[:n])
