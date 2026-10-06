import numpy as np


def mgf(values, probabilities, t):
    """
    values: 1D array-like of the outcomes X can take
    probabilities: 1D array-like, same length, non-negative, sums to 1
    t: float or NumPy array

    Returns:
        The moment generating function E[exp(t * X)] at t, with the same
        shape as t.
    """
    # TODO: Implement the weighted sum from Theory.
    values = np.asarray(values, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    t = np.asarray(t, dtype=float)
    total = (probabilities * np.exp(t[..., None] * values)).sum(axis=-1)
    return total if total.ndim else float(total)


def pgf(probabilities, z):
    """
    probabilities: 1D array-like where probabilities[k] = P(X = k)
    z: float or NumPy array

    Returns:
        The probability generating function E[z ** X] at z, with the same
        shape as z.
    """
    # TODO: Implement the series from Theory with the probabilities as
    # coefficients.
    probabilities = np.asarray(probabilities, dtype=float)
    z = np.asarray(z, dtype=float)
    powers = z[..., None] ** np.arange(len(probabilities))
    total = (probabilities * powers).sum(axis=-1)
    return total if total.ndim else float(total)


def mgf_moments(mgf_fn, step=1e-3):
    """
    mgf_fn: function taking a float t and returning a float M(t)
    step: the finite-difference step h

    Returns:
        (first_moment, second_moment) as floats: M'(0) and M''(0),
        estimated with central finite differences of size `step`.
    """
    # TODO: Differentiate numerically at t = 0, as in Theory.
    h = step
    first = (mgf_fn(h) - mgf_fn(-h)) / (2.0 * h)
    second = (mgf_fn(h) - 2.0 * mgf_fn(0.0) + mgf_fn(-h)) / h**2
    return float(first), float(second)


def pgf_mean_variance(probabilities):
    """
    probabilities: 1D array-like where probabilities[k] = P(X = k)

    Returns:
        (mean, variance) as floats, computed from the derivatives of the
        probability generating function at z = 1.
    """
    # TODO: Use G'(1), G''(1) and the variance identity from Theory.
    p = np.asarray(probabilities, dtype=float)
    k = np.arange(len(p))
    first_derivative = float(np.sum(k * p))
    second_derivative = float(np.sum(k * (k - 1) * p))
    variance = second_derivative + first_derivative - first_derivative**2
    return first_derivative, float(variance)



def sum_distribution(p, q):
    """
    p, q: probability lists for independent X and Y, where p[k] = P(X = k)
        and q[k] = P(Y = k)

    Returns:
        A 1D array of length len(p) + len(q) - 1 holding P(X + Y = n).
    """
    # TODO: Multiply the two generating functions, as in Theory.
    return np.convolve(np.asarray(p, dtype=float), np.asarray(q, dtype=float))
