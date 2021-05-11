from step_size import *
import matplotlib.pyplot as plt
from part_1 import find_a_hat


def projected_gradiant_descent(X_matrix, y, m, n, r, epsilon, iterations, step_size: Step_Size):
    a_0 = np.random.randn(n+1,)
    a_vec = [project_2_ball(a_0, r)]
    for t in range(iterations):
        a_t = a_vec[t]
        grad = calculating_problem_gradiant_by_a(X_matrix, y, a_t)
        if L_2_norm(grad) > epsilon:
            a_vec.append(project_2_ball(a_t - step_size.calc(grad) * grad, r))
        else:
            print("Reached Threshold")
            return a_vec
    print("Reached maximum number of iterations")
    return a_vec


def plot_error(a_hat, a_estimators, labels, colors, title, fig_path, r, m, X_matrix, y, x_axis_vec=None, xlabel="# Iterations", xlim=None):
    a_hat = project_2_ball(a_hat, r)
    h_star = h(X_matrix, y, a_hat, m)
    estimators_errors = [[h(X_matrix, y, a_est, m) - h_star for a_est in a_est] for a_est in a_estimators]

    plt.figure()
    x_axis_vec = [range(len(errors)) for errors in estimators_errors] if x_axis_vec is None else x_axis_vec
    for errors, label, color, x_axis in zip(estimators_errors, labels, colors, x_axis_vec):
        plt.semilogy(x_axis, errors[:len(x_axis)], label=label, color=color)
    plt.xlabel(xlabel)
    plt.ylabel("Estimation Error")
    plt.title(title)
    plt.legend(loc='best')
    if xlim is not None:
        plt.xlim(xlim)
    plt.savefig(fig_path)
    plt.close()


if __name__ == '__main__':
    m = Consts.M
    n = Consts.N
    A = Consts.A
    r = Consts.R

    X_matrix = create_X_matrix(generate_x_vector(m), n)
    f_X = calculating_f_X(X_matrix, A)
    Y = create_y_sample(f_X)
    L = calc_L(X_matrix, m)

    a_estimated_analytic = find_a_hat(X_matrix, Y)

    inverse_estimators = projected_gradiant_descent(X_matrix, Y, m, n, r, Consts.EPSILON, 1000, Step_Size_inverse_t(X_matrix, Y, m, r))
    adaGrad_estimators = projected_gradiant_descent(X_matrix, Y, m, n, r, Consts.EPSILON, 1000, Step_Size_AdaGrad(X_matrix, Y, m, r))

    plot_error(a_hat=a_estimated_analytic,
               a_estimators=[adaGrad_estimators, inverse_estimators],
               labels=['AdaGrad Error', 'Decreasing Learning Rate Error'],
               colors=['green', 'red'],
               title='AdaGrad LR Error Vs. Decreasing LR Error',
               fig_path="Adagrad_decrease_error.png",
               r=r, m=m, X_matrix=X_matrix, y=Y)

    constant_1_10L = projected_gradiant_descent(X_matrix, Y, m, n, r, Consts.EPSILON, 1000, Step_Size_Const(1 / (10 * L)))
    constant_1_L = projected_gradiant_descent(X_matrix, Y, m, n, r, Consts.EPSILON, 1000, Step_Size_Const(1 / L))
    constant_10_L = projected_gradiant_descent(X_matrix, Y, m, n, r, Consts.EPSILON, 1000, Step_Size_Const(10 / L))

    plot_error(a_hat=a_estimated_analytic,
               a_estimators=[constant_1_10L, constant_1_L, constant_10_L],
               labels=[r'$\eta_t = \frac{1}{10L}$', r'$\eta_t = \frac{1}{L}$', r'$\eta_t = \frac{10}{L}$'],
               colors=['yellow', 'green', "red"],
               title='Estimation error evaluations with constant step sizes',
               fig_path="estimation_error_of_constant_step_size.png",
               r=r, m=m, X_matrix=X_matrix, y=Y)

    plot_error(a_hat=a_estimated_analytic,
               a_estimators=[constant_1_L, adaGrad_estimators],
               labels=[r'$\eta_t = \frac{1}{L}$', 'AdaGrad Error'],
               colors=['red', 'green'],
               title=r'AdaGrad Vs. $\eta_t = \frac{1}{L}$ Step Size',
               fig_path="estimation_error_comparison_constant_adagrad_lr.png",
               r=r, m=m, X_matrix=X_matrix, y=Y)
