from model.himmelblau import himmelblau
from model.gradient_descent import GradientDescent, gradient
from model.quadratic_task import quadratic_function, quadratic_gradient
from model.quadratic_optimizer import QuadraticOptimizer

class OptimizerController:
    def __init__(self):
        self.optimizers = {
            "Gradient Descent": lambda step, it: GradientDescent(himmelblau, gradient, step, it),
            "Quadratic Task": lambda step, it: QuadraticOptimizer(quadratic_function, quadratic_gradient, step, it)
        }

    def get_optimizer(self, name, step_size, iterations):
        return self.optimizers[name](step_size, iterations)

    def get_available_optimizers(self):
        return list(self.optimizers.keys())