from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    def __init__(self, func, grad_func, step_size, iterations):
        self.func = func
        self.grad_func = grad_func
        self.step_size = step_size
        self.iterations = iterations

    @abstractmethod
    def optimize(self, x0, y0):
        pass