import numpy as np
from model.base_optimizer import BaseOptimizer
from model.quadratic_task import quadratic_function, quadratic_gradient, project_to_constraints

class QuadraticOptimizer(BaseOptimizer):
    def optimize(self, x0, y0):
        tolerance = 0.001
        points = np.zeros((self.iterations + 1, 2))
        current_point = np.array([x0, y0])
        points[0] = current_point
        actual_iter = 0

        for i in range(self.iterations):
            grad = quadratic_gradient(current_point[0], current_point[1])
            if np.linalg.norm(grad) < tolerance:
                break
            new_point = current_point - self.step_size * grad
            new_point[0], new_point[1] = project_to_constraints(new_point[0], new_point[1])
            points[i + 1] = new_point
            if np.linalg.norm(new_point - current_point) < tolerance:
                break
            current_point = new_point
            actual_iter = i + 1

        return points[:actual_iter + 1]