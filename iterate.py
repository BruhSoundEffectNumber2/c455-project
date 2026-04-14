import numpy as np


def iterate(l: np.float64, b: np.float64, x0: np.float64, lim: int) -> tuple[int, np.ndarray]:
    """Iterates Hassell's map the given number of times, returning the number of iterations and the array of iterates."""
    out = np.zeros(lim, dtype=np.float64)
    x = l * x0 / np.pow(1 + x0, b)
    out[0] = x

    # Start at 1 because we've already done the first iteration with x0
    for i in range(1, lim):
        # NanInf check
        if not np.isfinite(x):
            raise RuntimeError(f"x has become NaN or Inf at {i} steps")
        
        # Escape criteria
        if x > 100:
            return i, out

        x = l * x / np.pow(1 + x, b)
        out[i] = x

    return lim, out

def iterate_postrans(l: np.float64, b: np.float64, x0: np.float64, lim: int) -> tuple[int, np.ndarray]:
    """Iterates Hassell's map lim + 200 times, returning the number of iterations and the array of iterates after the first 200."""
    out = np.zeros(lim, dtype=np.float64)
    x = l * x0 / np.pow(1 + x0, b)

    # Start at 1 because we've already done the first iteration with x0
    for i in range(1, lim + 200):
        # NanInf check
        if not np.isfinite(x):
            raise RuntimeError(f"x has become NaN or Inf at {i} steps")
        
        # Escape criteria
        if x > 100:
            return i - 200, out

        x = l * x / np.pow(1 + x, b)
        if i >= 200:
            out[i - 200] = x

    return lim, out