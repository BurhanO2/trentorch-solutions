import numpy as np


def rank(A):
    """
    A: an (m, n) NumPy array

    Returns:
        The rank of A: the number of linearly independent rows,
        computed via row-reduction (skip a column with no usable pivot
        rather than treating it as an error).
    """
    # TODO: Implement row-reduction with a separate pivot-row/pivot-
    # column pointer, counting pivots found, from Theory.
    M = A.astype(float).copy()
    rows, cols = M.shape
    pivot_row = 0
    pivot_count = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        pivot = None
        for r in range(pivot_row, rows):
            if abs(M[r, col]) > TOLERANCE:
                pivot = r
                break
        if pivot is None:
            continue

        M[[pivot_row, pivot]] = M[[pivot, pivot_row]]

        for r in range(pivot_row + 1, rows):
            multiplier = M[r, col] / M[pivot_row, col]
            M[r] = M[r] - multiplier * M[pivot_row]

        pivot_row += 1
        pivot_count += 1

    return pivot_count


def nullity(A):
    """
    A: an (m, n) NumPy array

    Returns:
        The nullity of A: A.shape[1] - rank(A), per the Rank-Nullity
        Theorem.
    """
    # TODO: Implement using rank(A) from Theory's formula.
    return A.shape[1] - rank(A)
