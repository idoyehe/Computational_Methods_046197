import numpy as np
import matplotlib.pyplot as plt
from simulator import calc_f
from estimators import project_radious_r


def plot_estimation(x, f, y, a_estimated, title, fig_path):
    est_f = [calc_f(xi, a_estimated) for xi in x]
    indices = np.argsort(x)
    x_sorted = [x[i] for i in indices]
    y_sorted = [y[i] for i in indices]
    f_sorted = [f[i] for i in indices]
    est_f_sorted = [est_f[i] for i in indices]
    plt.figure()
    plt.scatter(x_sorted, y_sorted, label="y", alpha=0.5, s=8, color="none", edgecolors="green")
    plt.plot(x_sorted, f_sorted, label="f(x)", linewidth=3)
    plt.plot(x_sorted, est_f_sorted, label=r'$\hat{f}(x)$')
    plt.xlabel("x values")
    plt.ylabel("function value")
    plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig(fig_path)
    plt.close()


def plot_error(a_global_minima, a_estimators, labels, title, fig_path, r, m, X, y,
               x_axis_vec=None, xlabel="t [Iterations]", xlim=None):
    def h(X, y, a, m, esitmators):
        try:
            return (np.linalg.norm(y - X @ a) ** 2) / (2*m)
        except:
            breakpoint()
    a_star = project_radious_r(a_global_minima, r)
    ha_star = h(X, y, a_star, m, None)
    estimators_errors = [[np.linalg.norm(ha_star - h(X, y, a_est, m, a_estimators)) for a_est in a_est] for a_est in a_estimators]
    plt.figure()
    x_axis_vec = [range(len(errors)) for errors in estimators_errors] if x_axis_vec is None else x_axis_vec
    for errors, label, x_axis in zip(estimators_errors, labels, x_axis_vec):
        plt.semilogy(x_axis, errors[:len(x_axis)], label=label)
    plt.xlabel(xlabel)
    plt.ylabel("Estimation Error")
    plt.title(title)
    plt.legend(loc='best')
    if xlim is not None:
        plt.xlim(xlim)
    plt.savefig(fig_path)
    plt.close()
