import numpy as np
from matplotlib import pyplot as plt


class MinimizationProblem:
    def __init__(self, name: str, number_constraints):
        self.name: str = name
        self.number_constraints: int = number_constraints
        self.target_func = None
        self.dLdx = self.dLdy = self.ddLdxdy = self.ddLddx = self.ddLddy = None
        self.constrains = None
        self.starts_vector = None

    def gradient(self, x, y, t):
        return np.array([self.dLdx(x, y, t), self.dLdy(x, y, t)])

    def hessian(self, x, y, t):
        hess = np.array([[self.ddLddx(x, y, t), self.ddLdxdy(x, y, t)], [self.ddLdxdy(x, y, t), self.ddLddy(x, y, t)]])
        assert hess.shape == (2, 2)
        return hess


def solve_inner(current_x_vec, t, p:MinimizationProblem, num_iter=5):
    x_vec = current_x_vec
    for i in range(num_iter):
        x_vec = x_vec - 0.01 * np.linalg.inv(p.hessian(x_vec[0], x_vec[1], t)) @ p.gradient(x_vec[0], x_vec[1], t)
    return x_vec


def solve_outer(mu, t_0, epsilon, p:MinimizationProblem):
    x_vecs = [p.starts_vector]
    t = t_0
    while p.number_constraints / t > epsilon:
        next_vec = solve_inner(x_vecs[-1], t, p)
        x_vecs.append(next_vec)
        t *= mu
    return x_vecs


def plot_cont(p: MinimizationProblem, xlims, ylims, x_values, fig_path):
    x_grid, y_grid = np.meshgrid(np.linspace(xlims[0], xlims[1], n), np.linspace(ylims[0], ylims[1], n))
    constrain = p.constrains(x_grid, y_grid)
    z = p.target_func(x_grid, y_grid)
    z_w_contrains = z * constrain
    plt.figure()
    plt.contour(x_grid, y_grid, z_w_contrains, corner_mask=True, levels=n, zorder=1)
    plt.colorbar()
    for vec in x_values[:-1]:
        plt.scatter(vec[0], vec[1], color='black', zorder=2)
    plt.scatter(x_values[-1][0], x_values[-1][1], color='black', zorder=2, label=r'$x_t^*$ path')
    plt.title("Solving " + p.name + " with Newton's Method and Log Barrier Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc='upper right')
    plt.savefig(fig_path)


def init_p1():
    p_1_problem = MinimizationProblem("P1", 2)
    p_1_problem.constrains = lambda x, y: abs(x - 0.5 * y - 2) <= 1
    p_1_problem.target_func = lambda x, y: (x ** 2) + (y ** 2) + 5 * x - 10 * y
    p_1_problem.dLdx = lambda x, y, t: t * (2 * x + 5) - 1 / (x - 0.5 * y - 3) - 1 / (x - 0.5 * y - 1)
    p_1_problem.ddLddx = lambda x, y, t: 2 * t + 1 / ((x - 0.5 * y - 3) ** 2) + 1 / ((x - 0.5 * y - 1) ** 2)
    p_1_problem.dLdy = lambda x, y, t: t * (2 * y - 10) + 0.5 / (x - 0.5 * y - 3) + 0.5 / (x - 0.5 * y - 1)
    p_1_problem.ddLddy = lambda x, y, t: 2 * t + 0.25 / ((x - 0.5 * y - 3) ** 2) + 0.25 / ((x - 0.5 * y - 1) ** 2)
    p_1_problem.ddLdxdy = lambda x, y, t: -0.5 / ((x - 0.5 * y - 3) ** 2) - 0.5 / ((x - 0.5 * y - 1) ** 2)
    p_1_problem.starts_vector = np.array([2, 0])
    return p_1_problem


def init_p2():
    p_2_problem = MinimizationProblem("P2", 3)
    p_2_problem.constrains = lambda x, y: ((x - 1) ** 2 + (y - 2) ** 2 <= 9) * (x >= 2) * (y >= 0)
    p_2_problem.target_func = lambda x, y: ((x - 3) ** 2) + (y ** 2) + 3 * x - 2 * y
    p_2_problem.dLdx = lambda x, y, t: t * (2 * x - 3) + 2 * (x - 1) / (-((x - 1) ** 2) - ((y - 2) ** 2) + 9) - 1 / (x - 2)
    p_2_problem.dLdy = lambda x, y, t: t * (2 * y - 2) + 2 * (y - 2) / (-((x - 1) ** 2) - ((y - 2) ** 2) + 9) - 1 / y
    p_2_problem.ddLdxdy = lambda x, y, t: 4 * (x - 1) * (y - 2) / ((-((x - 1) ** 2) - ((y - 2) ** 2) + 9) ** 2)
    p_2_problem.ddLddx = lambda x, y, t: 2 * t + 2 / (-((x - 1) ** 2) - ((y - 2) ** 2) + 9) + 4 * ((x - 1) ** 2) / (
            (-((x - 1) ** 2) - ((y - 2) ** 2) + 9) ** 2) \
                                         + 1 / ((x - 2) ** 2)
    p_2_problem.ddLddy = lambda x, y, t: 2 * t + 2 / (-((x - 1) ** 2) - ((y - 2) ** 2) + 9) + 4 * ((y - 2) ** 2) / (
            (-((x - 1) ** 2) - ((y - 2) ** 2) + 9) ** 2) \
                                         + 1 / (y ** 2)
    p_2_problem.starts_vector = np.array([3, 1])

    return p_2_problem


if __name__ == '__main__':
    p_1 = init_p1()
    p_2 = init_p2()

    mu = 1.5
    t0 = 1
    epsilon = 1e-10
    n = 500
    x_values_p1 = solve_outer(mu, t0, epsilon, p_1)
    x_values_p2 = solve_outer(mu, t0, epsilon, p_2)
    plot_cont(p_1, xlims=[1, 3], ylims=[-2, 3], x_values=x_values_p1, fig_path="p1.png")
    plot_cont(p_2, xlims=[1.5, 4], ylims=[-0.5, 5], x_values=x_values_p2, fig_path="p2.png")
