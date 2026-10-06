import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from _load import load_solution

# _cond_prob = load_solution("00-math-and-statistics/03-probability/04-conditional-probability")
# marginal_x = _cond_prob.marginal_x
# marginal_y = _cond_prob.marginal_y

# kl_divergence = load_solution("00-math-and-statistics/04-information-theory/03-kl-divergence").kl_divergence

def kl_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """
    KL divergence measures exactly the GAP Gibbs' inequality guarantees
    is non-negative: how many EXTRA bits cross-entropy costs, beyond
    the true distribution's own entropy.

        KL(p || q) = H(p, q) - H(p)

    `entropy` and `cross_entropy` are already provided above, this is
    a one-line combination of them, not a new formula to derive.
    """
    return cross_entropy(p, q) - entropy(p)

def marginal_x(joint: np.ndarray) -> np.ndarray:
    """
    `joint` is a 2D array where joint[i, j] = P(X=i, Y=j). The marginal
    distribution of X alone, P(X=i), sums out every value of Y:

        P(X=i) = sum_j(P(X=i, Y=j))
    """
    return np.sum(joint, axis=1)    


def marginal_y(joint: np.ndarray) -> np.ndarray:
    """
    Same idea as marginal_x, summed the other way: P(Y=j) sums out
    every value of X.
    """
    return np.sum(joint, axis=0) 



def mutual_information(joint: np.ndarray, base: float = 2.0) -> float:
    """
    Mutual information measures how far the actual joint distribution
    is from what it WOULD look like if X and Y were independent:

        MI(X; Y) = KL(P(X, Y) || P(X) * P(Y))

    Build the independent joint (the outer product of the two
    marginals, 03-probability/04-conditional-probability's own
    marginal_x/marginal_y are already provided above), then measure
    how far the real joint diverges from it, via KL divergence
    (already provided above too) on the two flattened tables.
    """
    p_x = marginal_x(joint)
    p_y = marginal_y(joint)
    independent_join = np.outer(p_x, p_y)
    return kl_divergence(joint.flatten(), independent_join.flatten(), base)
