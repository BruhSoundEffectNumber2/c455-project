import numpy as np

BETA = 10

def h(l: np.float64, x: np.float64) -> np.float64:
    """Hassell's map function."""
    return l * x / np.pow(1 + x, BETA)

def hprime(l: np.float64, x: np.float64):
    """Derivative of Hassell's map function."""
    return l * (1 + x) ** (-BETA - 1) * (1 + x - BETA * x)

def h_fixed_points(l: np.float64) -> np.float64:
    """Returns the second fixed point of Hassell's map. Only valid for l>=1. At l=1, there is only one fixed point at 0."""
    if l < 1:
        return np.nan
    
    return np.power(l, 1 / BETA) - 1


def iterate(l: np.float64, x0: np.float64, lim: int) -> tuple[int, np.ndarray]:
    """Iterates Hassell's map the given number of times, returning the number of iterations and the array of iterates."""
    out = np.zeros(lim, dtype=np.float64)
    x = h(l, x0)
    out[0] = x

    # Start at 1 because we've already done the first iteration with x0
    for i in range(1, lim):
        # NanInf check
        if not np.isfinite(x):
            raise RuntimeError(f"x has become NaN or Inf at {i} steps")
        
        # Escape criteria
        if x > 100:
            return i, out

        x = h(l, x)
        out[i] = x

    return lim, out

def iterate_postrans(l: np.float64, x0: np.float64, lim: int) -> tuple[int, np.ndarray]:
    """Iterates Hassell's map lim + 200 times, returning the number of iterations and the array of iterates after the first 200."""
    out = np.zeros(lim, dtype=np.float64)
    x = h(l, x0)

    # Start at 1 because we've already done the first iteration with x0
    for i in range(1, lim + 200):
        # NanInf check
        if not np.isfinite(x):
            raise RuntimeError(f"x has become NaN or Inf at {i} steps")
        
        # Escape criteria
        if x > 100:
            return i - 200, out

        x = h(l, x)
        if i >= 200:
            out[i - 200] = x

    return lim, out

def find_super_stable(k: np.int64, l_guess: np.float64) -> tuple[np.int64, np.float64]:
    n = 2 ** (k - 1)
    l = l_guess

    for i in range(1000):
        x = np.float64(1.0) / (BETA - 1)
        xp = np.float64(0)

        for _ in range(n):
            xp = x / np.pow((x + 1), BETA)
            x = h(l, x)

        corr = (x - 1 / (BETA - 1)) / xp
        l -= corr

        if not np.isfinite(l):
            raise RuntimeError(f"l has become NaN or Inf at {i} iterations")

        # print(l)

        if abs(corr) < 1e-14:
            return i + 1, l
    
    raise RuntimeError(f"Failed to find super stable point (last guess: {l})")