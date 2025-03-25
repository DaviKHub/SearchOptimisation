import numpy as np
from numba import jit

@jit(nopython=True)
def quadratic_function(x, y):
    return 2 * x**2 + 4 * x * y - 6 * x - 3 * y

@jit(nopython=True)
def quadratic_gradient(x, y):
    df_dx = 4 * x + 4 * y - 6
    df_dy = 4 * x - 3
    return np.array([df_dx, df_dy])

def is_within_constraints(x, y):
    return x >= 0 and y >= 0 and (x + y) <= 1 and (2 * x + 3 * y) <= 4

def project_to_constraints(x, y):
    x = max(0, min(x, 1))
    y = max(0, min(y, 1))
    if (x + y) > 1:
        x, y = x / (x + y), y / (x + y)
    if (2 * x + 3 * y) > 4:
        scale = 4 / (2 * x + 3 * y)
        x, y = x * scale, y * scale
    return x, y