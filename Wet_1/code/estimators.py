import numpy as np
import time


def estimate_a_analytic(X, y):
    a_hat = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
    return a_hat


def project_radious_r(a, r):
    if np.linalg.norm(a) < r:
        return a
    return r * a / np.linalg.norm(a)


def estimate_a_constraint_GD(X, y, lr_handler, r=4, m=1e4, N=3, max_iter=500, eps=5e-4, constant_lr=None):
    def grad_h(m, y, X, a):
        return - np.transpose(X) @ (y - X @ a) / m

    lr_hand = lr_handler(X=X, r=r, y=y, m=m, constant_value=constant_lr)
    a_vec = [np.random.randn(N+1,)]
    for i in range(max_iter):
        grad = grad_h(m, y, X, a_vec[-1])
        if np.linalg.norm(grad) < eps:
            return a_vec
        curr_lr = lr_hand.get_lr(grad)
        a_vec.append(project_radious_r(a=a_vec[-1] - curr_lr * grad, r=r))
    print("Reached maximum number of iterations")
    return a_vec


def estimate_a_constraint_SGD(X, y, lr_handler, b, r=4, m=1e4, N=3, max_iter=500,
                              constant_lr=None, stop_time=np.inf):
    def grad_1_samp(y_sample, x_sample, a):
        return - np.transpose(x_sample) * (y_sample - x_sample @ a)
    lr_hand = lr_handler(X=X, r=r, y=y, m=m, constant_value=constant_lr)
    a_vec = [np.random.randn(N+1,)]
    init_time = time.time()
    time_vec = [0]
    for i in range(max_iter):
        grad = 0
        for sample_ind in np.random.randint(m, size=b):
            grad += grad_1_samp(y[sample_ind], X[sample_ind, :], a_vec[-1]) / b
        curr_lr = lr_hand.get_lr(grad)
        a_vec.append(project_radious_r(a=a_vec[-1] - curr_lr * grad, r=r))
        time_vec.append(time.time() - init_time)
        if time_vec[-1] > stop_time:
            break
    print("Reached maximum number of iterations")
    return time_vec, a_vec
