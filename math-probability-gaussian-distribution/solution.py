import numpy as np


def gaussian_pdf(x, mu, sigma):
    """
    x:     scalar or NumPy array of any shape
    mu:    the mean
    sigma: the standard deviation (> 0)

    Returns:
        the Gaussian probability density at x, same shape as x.
    """
    # TODO: Implement the closed-form Gaussian PDF from Theory.
    x = np.array(x, dtype=float)
    coeff = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - mu) ** 2) / (2 * sigma**2)
    return coeff * np.exp(exponent)


def sample_gaussian(mu, sigma, n, uniform_draws):
    """
    mu, sigma:     Gaussian parameters
    n:             number of samples to return
    uniform_draws: exactly 2 * n floats in [0, 1), pre-generated

    Returns:
        n Gaussian-distributed samples, as a flat array.
    """
    # TODO: Implement Box-Muller from Theory. Consume uniform_draws in
    # pairs (u1, u2) to produce standard-normal samples in pairs (z0, z1),
    # then shift/scale by mu and sigma. Only slice down to n at the end.
    u = np.array(uniform_draws, dtype=float)
    u1, u2 = u[0::2], u[1::2]
    u1 = np.clip(u1, 1e-12, None)

    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    z0 = r * np.cos(theta)
    z1 = r * np.sin(theta)

    z = np.empty(2 * len(u1))
    z[0::2], z[1::2] = z0, z1

    return mu + sigma * z[:n]
