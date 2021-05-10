import numpy as np

from utils import *


class Step_Size(object):
    def calc(self, grad):
        pass


class Step_Size_Const(Step_Size):
    def __init__(self, const):
        super(Step_Size_Const, self).__init__()
        self._const = const

    def calc(self, _):
        return self._const


class Step_Size_Per_t(Step_Size):
    def __init__(self, X_matrix, y, m, r):
        super(Step_Size_Per_t, self).__init__()
        self.current_iteration = 0
        self.D = 2 * r
        max_lambda = max(np.linalg.eigvals(np.matmul(np.transpose(X_matrix), X_matrix) / m))
        self.G = 2 * r * max_lambda + L_2_norm(np.matmul(np.transpose(X_matrix), y) / m)


class Step_Size_inverse_t(Step_Size_Per_t):
    def __init__(self, x, y, m, r):
        super(Step_Size_inverse_t, self).__init__(x, y, m, r)

    def calc(self, _):
        self.current_iteration += 1
        return self.D / (self.G * np.sqrt(self.current_iteration))


class Step_Size_AdaGrad(Step_Size_Per_t):
    def __init__(self, x, y, m, r):
        super(Step_Size_AdaGrad, self).__init__(x, y, m, r)
        self._gradients_L2_history = []

    def calc(self, grad):
        self.current_iteration += 1
        self._gradients_L2_history.append(L_2_norm(grad) ** 2)
        return self.D / (2 * np.sqrt(sum(self._gradients_L2_history)))
