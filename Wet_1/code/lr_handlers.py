import numpy as np


class LrHandler:
    def __init__(self, X, r, y, m, constant_value):
        raise ValueError("Virtual")

    def get_lr(self, grad):
        raise ValueError("Virtual")


class LrHandlerDecreasing(LrHandler):
    def __init__(self, X, r, y, m, constant_value=None):
        self.G = (r * np.max(np.linalg.eigvals(np.transpose(X) @ X)) + np.linalg.norm(np.transpose(X) @ y))/m
        self.t = 0
        self.D = 2*r

    def get_lr(self, grad):
        self.t += 1
        return self.D/(self.G * np.sqrt(self.t))


class LrHandlerAdaGrad(LrHandler):
    def __init__(self, r, X=None, y=None, m=None, constant_value=None):
        self.D = 2*r
        self.sum_grads = 0

    def get_lr(self, grad):
        self.sum_grads += grad ** 2
        return self.D/np.sqrt(2 * self.sum_grads)


class LrHandlerConstant(LrHandler):
    def __init__(self, constant_value, X=None, r=None, y=None, m=None):
        self.lr = constant_value

    def get_lr(self, grad=None):
        return self.lr

