from numba import jit
from numpy import pi, exp, ndarray, gradient, concatenate


__all__ = ['gauss', 'ispeak']


@jit
def gauss(x: float, sigma: float) -> float:
    return 1 / (2*pi)**0.5 * exp(-(x/sigma)**2/2)


@jit
def ispeak(arr: ndarray) -> ndarray:
    diff = gradient(arr)  # shape: k
    ispos = 0 < diff  # shape: k
    return concatenate([[False], ispos[:-1] & ~ispos[1:]])  # shape: k
