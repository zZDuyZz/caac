from scipy.stats import binom
DELTA = 0.10
def fails(k, n):
    return bool(k > DELTA * n and binom.sf(k - 1, n, DELTA) < 0.01)
