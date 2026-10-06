import numpy as np


def is_positive_semidefinite(a: np.ndarray, tol: float = 1e-10) -> bool:
    """
    a: a 2D array
    tol: eigenvalues down to -tol count as zero

    Returns:
        True if a is square, symmetric (np.allclose) and has no
        eigenvalue below -tol, otherwise False.
    """
    # TODO: Loosen the strict eigenvalue test from 01-positive-definite.
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.allclose(a, a.T):
        return False
    return bool(np.linalg.eigvalsh(a)[0] >= -tol)


def gram_matrix(x: np.ndarray) -> np.ndarray:
    """
    x: m x d array whose rows are vectors

    Returns:
        The m x m matrix of all pairwise dot products of the rows of x.
    """
    # TODO: Build the Gram matrix from Theory.
    x = np.asarray(x, dtype=float)
    return x @ x.T


def nearest_psd(a: np.ndarray) -> np.ndarray:
    """
    a: a square 2D array

    Returns:
        The symmetric positive semi-definite matrix closest to
        (a + a.T) / 2 in Frobenius norm, same shape as a. a must not be
        modified.
    """
    # TODO: Clip the negative eigenvalues of the symmetric part, as in Theory.
    a = np.asarray(a, dtype=float)
    symmetric = (a + a.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, 0.0, None)
    result = (eigenvectors * clipped) @ eigenvectors.T
    return (result + result.T) / 2.0
