import numpy as np

np.random.seed(47)


class Consts(object):
    N = 3
    M = 10000
    A = np.array([0, 1, 0.5, -2])
    R = 4
    EPSILON = 0.001


def generate_x_vector(m: int = Consts.M):
    return np.random.uniform(low=-1.0, high=1.0, size=m)


def calculating_f_X(X_matrix, A):
    return X_matrix @ A


def calculating_problem_gradiant_by_a(X_matrix, y, a):
    m = len(y)
    X_T_X = np.matmul(np.transpose(X_matrix), X_matrix)
    return (1 / m) * (-np.matmul(np.transpose(X_matrix), y) + np.matmul(X_T_X, a))


def create_y_sample(f_X):
    m = len(f_X)
    noise = np.random.normal(loc=0, scale=np.sqrt(0.5), size=m)
    return f_X + noise


def create_X_matrix(X, n):
    return np.array([[x_i ** j for j in range(n + 1)] for x_i in X])


def L_2_norm(vector):
    return np.linalg.norm(vector, ord=2)


def in_ball(vector, r):
    return L_2_norm(vector) <= r


def project_2_ball(v, r):
    L_2_v = L_2_norm(v)
    if L_2_v <= r:
        return v
    else:
        return v * (r / L_2_v)


def h(X, y, a, m):
    return (1 / (2 * m)) * (L_2_norm(y - np.matmul(X, a)) ** 2)


def calc_L(X_matrix, m):
    return max(np.linalg.eigvals(np.matmul(np.transpose(X_matrix), X_matrix) / m))
