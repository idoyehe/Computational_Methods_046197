import random

import numpy as np
np.random.seed(47)
random.seed(47)

class Consts(object):
    N = 3
    M = 10000
    A = np.array([0, 1, 0.5, -2])
    R = 4
    EPSILON = 0.001


def generate_x_vector(m: int = Consts.M):
    return np.array([np.random.uniform(low=-1, high=1.0) for _ in range(m)])


def create_X_matrix(X, n):
    return np.array([[x_i ** j for j in range(n + 1)] for x_i in X])


def calculating_f_X_matrix(X_matrix, A):
    return X_matrix @ A


def calculating_problem_gradiant_by_a(X_matrix, y, a):
    m = len(y)
    X_T_X = np.transpose(X_matrix) @ X_matrix
    return (1 / m) * (-np.transpose(X_matrix) @ y + X_T_X @ a)


def calculating_sample_gradiant_by_a(X_sample, y_sample, a):
    return - np.transpose(X_sample) * (y_sample - X_sample @ a)


def create_y_sample(f_X):
    m = len(f_X)
    mu, sigma = 0, np.sqrt(0.5)
    y = []
    for i in range(m):
        y.append(f_X[i] + np.random.normal(loc=mu, scale=sigma))
    return np.array(y)


def L_2_norm(vector):
    return np.linalg.norm(vector)


def in_ball(vector, r):
    return L_2_norm(vector) <= r


def project_2_ball(a, r):
    l_2_norm = L_2_norm(a)
    if in_ball(a, r):
        return a
    return r * a / l_2_norm


def h(X, y, a, m):
    return (1 / (2 * m)) * (L_2_norm(y - np.matmul(X, a)) ** 2)


def calc_L(X_matrix, m):
    return np.max(np.linalg.eigvals(np.transpose(X_matrix) @ X_matrix)) / m
