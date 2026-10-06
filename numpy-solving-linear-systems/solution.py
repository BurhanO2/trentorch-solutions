import numpy as np


def solve_system(coefficients: np.ndarray, constants: np.ndarray) -> np.ndarray:
    """
    Solve the linear system coefficients @ x = constants for x,
    using np.linalg.solve.
    """
    return np.linalg.solve(coefficients, constants)


def solve_via_inverse(coefficients: np.ndarray, constants: np.ndarray) -> np.ndarray:
    """
    Solve the same system as solve_system, but by explicitly
    computing the inverse of `coefficients` and multiplying it by
    `constants`. This exists to be compared against solve_system.
    """
    return np.linalg.inv(coefficients) @ constants


def solutions_agree(coefficients: np.ndarray, constants: np.ndarray) -> bool:
    """
    Solve the system both ways (using solve_system and
    solve_via_inverse) and return True if the two solutions are
    approximately equal (use np.allclose), False otherwise.
    """
    return bool(
        np.allclose(
            solve_system(coefficients, constants),
            solve_via_inverse(coefficients, constants),
        )
    )
