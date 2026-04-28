"""
Newton's method for super-stable parameters of the Hassell map.

    F(x) = lambda * x / (1 + x)^b,    x >= 0,  lambda > 0,  b > 1.

We fix b (default b = 6) and use lambda as the bifurcation parameter.
The critical point x0 = 1/(b-1) is independent of lambda, so a super-stable
cycle of period N = 2^(k-1) is a zero of

    g(lambda) = F^N(x0) - x0.

Newton's method on g uses both g(lambda) and g'(lambda), each computed by
iterating the orbit and a parallel "tangent" derivative recursion.

DERIVATIONS (verify these by hand before trusting the code).

    Critical point:        x0 = 1/(b-1).
    Period-1 super-stable: s1 = (b/(b-1))^b.
    For b=6: x0 = 1/5,  s1 = (6/5)^6 = 2.985984 exactly.

    Forward orbit:   x_{j+1} = lambda * x_j / (1+x_j)^b.
    Tangent (d/d lambda):
       x'_0 = 0
       x'_{j+1} = (x_j / (1+x_j)^b)                          [partial of F wrt lambda]
                + (lambda * (1 + (1-b)*x_j) / (1+x_j)^(b+1)) [partial of F wrt x]
                  * x'_j

The Newton step is then
    lambda_new = lambda - g(lambda) / g'(lambda).

SAFEGUARDS the loop adds, beyond textbook Newton:

  (1) Step cap.  |Delta lambda| is bounded by `step_cap` to prevent the
      iterate from leaving the basin of attraction.
  (2) Backtracking on blow-up.  If a tentative step produces a divergent
      orbit, the step is halved and retried.
  (3) Convergence on |Delta lambda|, not |g|.  For large N, |g| can be
      tiny even at non-roots; the honest measure is the size of the
      Newton correction itself.

USAGE.  Run as a script to reproduce s_1, ..., s_10 and the table of
Feigenbaum ratios delta_k.

    python3 hassell_newton.py
"""

import math


# -------- map and its derivative recursion --------

def F(x, lam, b):
    """Hassell map F(x) = lambda * x / (1+x)^b."""
    return lam * x / (1.0 + x)**b


def dF_dx(x, lam, b):
    """Partial derivative of F with respect to x at fixed lambda."""
    return lam * (1.0 + (1.0 - b) * x) / (1.0 + x)**(b + 1.0)


def critical_orbit(lam, N, b):
    """
    Iterate the critical orbit N times, simultaneously propagating the
    derivative with respect to lambda.

    Returns (x_N, x'_N) where x'_N = d x_N / d lambda.

    If the orbit blows up, returns (inf, inf) so the caller can backtrack.
    """
    x = 1.0 / (b - 1.0)   # critical point x0
    xp = 0.0              # x'_0  = 0  (x0 does not depend on lambda)
    for _ in range(N):
        if not math.isfinite(x) or x < 0 or x > 1e15:
            return float('inf'), float('inf')
        # IMPORTANT: update derivative first, using the CURRENT x_j
        partial_lam = x / (1.0 + x)**b
        xp_next = partial_lam + dF_dx(x, lam, b) * xp
        x = F(x, lam, b)
        xp = xp_next
    return x, xp


# -------- Newton solver with safeguards --------

def newton_super_stable(k, lam_guess,
                        b=6.0, step_cap=0.5, tol=1e-13, max_iter=200,
                        verbose=False):
    """
    Find lambda such that the critical orbit has a super-stable cycle of
    period N = 2^(k-1).

    Parameters
    ----------
    k          : integer >= 1.  Period is N = 2^(k-1).
    lam_guess  : initial guess for lambda.  Should be within ~1 of the truth.
    b          : map parameter (held fixed).  Default 6.0.
    step_cap   : maximum allowed |Delta lambda| per Newton step.
    tol        : convergence tolerance on |Delta lambda|.
    max_iter   : iteration limit.
    verbose    : if True, print one line per iteration.

    Returns lambda (or None on failure).
    """
    N = 2**(k - 1)
    x0 = 1.0 / (b - 1.0)
    lam = lam_guess

    for it in range(max_iter):
        xN, xpN = critical_orbit(lam, N, b)

        if not math.isfinite(xN):
            # Orbit blew up at current lambda -- pull back toward seed.
            lam = 0.5 * (lam + lam_guess)
            if verbose:
                print(f"  iter {it}: blow-up; retreat to lam = {lam:.6f}")
            continue

        g = xN - x0
        gp = xpN
        if abs(gp) < 1e-300:
            if verbose:
                print(f"  iter {it}: g'(lambda) ~ 0, abort")
            return None

        # Raw Newton correction, capped.
        delta = g / gp
        if abs(delta) > step_cap:
            delta = math.copysign(step_cap, delta)

        # Try the step; backtrack if it produces a divergent or wildly
        # worse orbit.
        damping = 1.0
        accepted = False
        for _ in range(20):
            lam_try = lam - damping * delta
            xN_try, _ = critical_orbit(lam_try, N, b)
            if math.isfinite(xN_try) and abs(xN_try - x0) < 10 * (abs(g) + 1.0):
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            if verbose:
                print(f"  iter {it}: backtracking failed at lam = {lam}")
            return None

        if verbose:
            print(f"  iter {it}: lam = {lam:.12f}   g = {g: .3e}   "
                  f"step = {-damping*delta: .3e}")

        lam = lam - damping * delta
        if abs(damping * delta) < tol:
            return lam

    return lam  # may not have fully converged


# -------- bootstrap s_1 ... s_K and Feigenbaum ratios --------

def feigenbaum_table(K=10, b=6.0,
                     s2_seed=17.0, s3_seed=32.0, s3_step_cap=1.0):
    """
    Compute s_1, s_2, ..., s_K for the Hassell map at fixed b, and the
    associated Feigenbaum ratios delta_k = (s_{k-1} - s_{k-2}) / (s_k - s_{k-1}).

    s_1 is the analytic value (b/(b-1))^b.
    s_2 and s_3 are seeded with caller-supplied visual estimates.
    s_4 and beyond are bootstrapped using the geometric scaling
        s_{k+1} ~ s_k + (s_k - s_{k-1}) / delta_current.

    Returns a list s with s[k] = s_k (1-indexed; s[0] is None).
    """
    x0 = 1.0 / (b - 1.0)

    s = [None]
    s.append((1.0 + x0)**b)                                          # s_1
    s.append(newton_super_stable(2, s2_seed, b=b))                   # s_2
    s.append(newton_super_stable(3, s3_seed, b=b, step_cap=s3_step_cap))  # s_3

    delta = 4.0  # rough starting estimate for the Feigenbaum constant
    for k in range(3, K):
        seed = s[k] + (s[k] - s[k-1]) / delta
        cap = max(0.05, 1.0 / 2**(k - 3))   # tighter cap for larger k
        nxt = newton_super_stable(k+1, seed, b=b, step_cap=cap, max_iter=400)
        if nxt is None:
            print(f"  bootstrap failed at k+1 = {k+1}")
            break
        s.append(nxt)
        delta = (s[k] - s[k-1]) / (s[k+1] - s[k])

    return s


def print_feigenbaum_table(s):
    print(f"{'k':>3} {'s_k':>20} {'s_k - s_{k-1}':>22} {'delta_k':>14}")
    print("-" * 64)
    for k in range(1, len(s)):
        sk = s[k]
        if k == 1:
            print(f"{k:>3} {sk:>20.13f} {'':>22} {'':>14}")
        elif k == 2:
            diff = sk - s[k-1]
            print(f"{k:>3} {sk:>20.13f} {diff:>22.13f} {'':>14}")
        else:
            diff = sk - s[k-1]
            prev_diff = s[k-1] - s[k-2]
            delta_k = prev_diff / diff
            print(f"{k:>3} {sk:>20.13f} {diff:>22.13f} {delta_k:>14.10f}")


# -------- main --------

if __name__ == "__main__":
    b = 6.0
    x0 = 1.0 / (b - 1.0)

    # Sanity test on s_1: must match the analytic value (b/(b-1))^b.
    print("=== sanity test: s_1 ===")
    s1_numeric = newton_super_stable(1, 3.0, b=b, verbose=True)
    s1_analytic = (1.0 + x0)**b
    print(f"  Newton:    s1 = {s1_numeric:.13f}")
    print(f"  analytic:  s1 = {s1_analytic:.13f}")
    print(f"  agreement: {abs(s1_numeric - s1_analytic):.2e}")
    print()

    # Full bootstrap.
    print("=== Feigenbaum bootstrap, k = 1 to 10 ===")
    s = feigenbaum_table(K=10, b=b)
    print()
    print_feigenbaum_table(s)
    print()

    # Estimate the cascade endpoint lambda_infinity.
    delta_final = (s[-2] - s[-3]) / (s[-1] - s[-2])
    lam_inf = s[-1] + (s[-1] - s[-2]) / (delta_final - 1.0)
    print(f"final delta estimate     = {delta_final:.10f}")
    print(f"true Feigenbaum constant = 4.6692016091...")
    print(f"lambda_infinity estimate = {lam_inf:.10f}")
