import matplotlib.pyplot as plt
from utils import *

def find_a_hat(X_matrix, Y):
    X_T_X = np.matmul(np.transpose(X_matrix), X_matrix)
    inv_X_T_X = np.linalg.inv(X_T_X)
    temp = np.matmul(inv_X_T_X, np.transpose(X_matrix))
    a_hat = np.matmul(temp, Y)
    return a_hat


def plot_estimation(X, f_X, Y, est_f_X, title, fig_path):
    indices = np.argsort(X)
    x_sorted = [X[i] for i in indices]
    y_sorted = [Y[i] for i in indices]
    f_sorted = [f_X[i] for i in indices]
    est_f_X_sorted = [est_f_X[i] for i in indices]
    plt.figure()
    plt.scatter(x_sorted, y_sorted, label="y", alpha=0.5, s=8, color="none", edgecolors="black")
    plt.plot(x_sorted, f_sorted, label="f(x)", linewidth=3)
    plt.plot(x_sorted, est_f_X_sorted, label=r'$\hat{f}(x)$')
    plt.xlabel("x values")
    plt.ylabel("function value")
    plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig(fig_path)
    plt.close()


if __name__ == '__main__':
    m = Consts.M
    n = Consts.N
    A = Consts.A

    X = generate_x_vector(m)
    X_matrix = create_X_matrix(X,n)
    f_X = calculating_f_X(X_matrix, A)
    Y = create_y_sample(f_X)
    a_hat = find_a_hat(X_matrix, Y)

    est_f_X = calculating_f_X(X_matrix, a_hat)

    plot_estimation(X, f_X, Y, est_f_X, "Least Square Estimator Vs. Real","Estimator_Vs_Real.png")
