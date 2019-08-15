"""Confirm the expressions below.

Define Gaussian function as follows:
    gauss(t, sigma) = 1 / sqrt(2 * pi) / sigma * exp(-(t / sigma)**2 / 2).

(1) The amp' and sigma' which satisfy the following expression:
    (amp' * gauss(t, sigma'))**2 == amp * gauss(t, sigma)
are:
    sigma' = sqrt(2) * sigma,
    amp' = root(8 * pi * amp**2 * sigma**2, 4).

(2) The electric field in time domain:
    E(t) = amp' * sin(k0 * t) * gauss(t, sigma'),
    sigma' = sqrt(2) * sigma,
    amp' = root(8 * pi * amp**2 * sigma**2, 4),
of which intensity envelope is:
    Envelope of E**2(t) = amp * gauss(t, sigma).
The field in frequency domain, E'(k) is approximately:
    E'(k) ~ amp' / 2j / sigma' * gauss(k - k0, 1 / sigma'),
and the intensity envelope is:
    abs(E')**2(k) ~ amp / 4 * gauss(k - k0, 1 / 2 / sigma).
"""
#%%
from sympy import *


#%% Confirm Expr (1)
t = symbols("t", real=True)
sigma, amp = symbols("sigma a", positive=True)

gauss = 1 / sqrt(2 * pi) / sigma * exp(-(t / sigma)**2 / 2)
expr0 = (root(8 * pi * amp**2 * sigma**2, 4)
         * gauss.subs(sigma, sqrt(2) * sigma))**2
expr1 = amp * gauss

simplify(expr0 - expr1) == 0


#%% Confirm Expr (2)
sigma_prime = sqrt(2) * sigma
amp_prime = root(8 * pi * amp**2 * sigma**2, 4)

expr2 = (amp_prime / 2 / sigma_prime
         * gauss.subs(sigma, 1 / sigma_prime))**2
expr3 = amp / 4 * gauss.subs(sigma, 1 / sigma / 2)

simplify(expr2 - expr3) == 0
