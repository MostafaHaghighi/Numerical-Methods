# -*- coding: utf-8 -*-
"""
@author: Mostafa Haghighi
Email: Mostafahaghighi.ce@gmail.com

"""
def secant(f, x0, x1, tol=1e-6, max_iter=100):
    """Find a root of `f(x) = 0` using the secant method.

    Args:
        f: Function to find the root of.
        x0, x1: Initial guesses.
        tol: Tolerance (stopping criterion).
        max_iter: Maximum number of iterations.

    Returns:
        Approximate root `x` where `f(x) ≈ 0`.
    """
    for _ in range(max_iter):
        fx1 = f(x1)
        if abs(fx1) < tol:
            return x1
        fx0 = f(x0)
        x_next = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        x0, x1 = x1, x_next
    return x1  # Return best guess if max_iter reached
#from secant import secant
import numpy as np

def fs(x, yvals, s):
    q = 1.5
    y0, y1 = yvals
    f0 = y1
    f1 = (2 * q * np.cos(2 * x) - s) * y0
    return np.array([f0, f1])

def rk4_gen_eig(fs, a, b, n, yinits, s):
    h = (b - a) / (n - 1)
    xs = a + np.arange(n) * h
    ys = np.zeros((n, yinits.size))
    yvals = np.copy(yinits)
    for j, x in enumerate(xs):
        ys[j, :] = yvals
        k0 = h * fs(x, yvals, s)
        k1 = h * fs(x + h / 2, yvals + k0 / 2, s)
        k2 = h * fs(x + h / 2, yvals + k1 / 2, s)
        k3 = h * fs(x + h, yvals + k2, s)
        yvals += (k0 + 2 * k1 + 2 * k2 + k3) / 6
    return xs, ys

def shoot(s):
    a, b, n = 0, 2 * np.pi, 500
    yinits = np.array([0., 5.])
    xs, ys = rk4_gen_eig(fs, a, b, n, yinits, s)
    wfinal = 0.
    return ys[-1, 0] - wfinal

if __name__ == '__main__':
    for sinit in (-0.4, 3.3, 8.5):
        sval = secant(shoot, sinit, sinit + 0.5)
        print(sval, end=" ")
