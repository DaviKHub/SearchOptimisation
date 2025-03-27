import numpy as np
from model.base_optimizer import BaseOptimizer

class GradientDescent(BaseOptimizer):
    def optimize(self, x0, y0):
        path = [(x0, y0, self.func(x0, y0))]
        for _ in range(self.iterations):
            grad = self.grad_func(x0, y0)
            x0 -= self.step_size * grad[0]
            y0 -= self.step_size * grad[1]
            z = self.func(x0, y0)
            print(f"{_}: ({x0:.4f}, {y0:.4f}, {z:.4f})")
            path.append((x0, y0, z))

            path.append((x0, y0, self.func(x0, y0)))
        return np.array(path)

def gradient(x, y):
    df_dx = 4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7)
    df_dy = 2 * (x**2 + y - 11) + 4 * y * (x + y**2 - 7)
    return np.array([df_dx, df_dy])