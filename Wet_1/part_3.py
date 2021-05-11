from step_size import *
import matplotlib.pyplot as plt
from part_1 import find_a_hat
from part_2 import plot_error


def projected_stochastic_gradiant_descent(X_matrix, y, m, n, r, iterations, step_size: Step_Size, batch_size):
    a_0 = np.random.randn(n + 1, )
    a_vec = [project_2_ball(a_0, r)]
    for t in range(iterations):
        a_t = a_vec[t]
        grad = 0
        for sample in np.random.randint(low=0, high=m, size=batch_size):
            grad += calculating_sample_gradiant_by_a(X_matrix[sample], y[sample], a_t)
        grad /= batch_size
        a_vec.append(project_2_ball(a_t - step_size.calc(grad) * grad, r))

    print("Reached maximum number of iterations")
    return a_vec


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

    inverse_estimators_1 = projected_stochastic_gradiant_descent(X_matrix, Y, m, n, r, 1000, Step_Size_inverse_t(X_matrix, Y, m, r),
                                                                 1)
    inverse_estimators_10 = projected_stochastic_gradiant_descent(X_matrix, Y, m, n, r, 1000, Step_Size_inverse_t(X_matrix, Y, m, r),
                                                                  10)
    inverse_estimators_100 = projected_stochastic_gradiant_descent(X_matrix, Y, m, n, r, 1000, Step_Size_inverse_t(X_matrix, Y, m, r),
                                                                   100)
    inverse_estimators_10000 = projected_stochastic_gradiant_descent(X_matrix, Y, m, n, r, 1000, Step_Size_inverse_t(X_matrix, Y, m, r), 10000)

    plot_error(a_hat=a_estimated_analytic,
               a_estimators=[inverse_estimators_1, inverse_estimators_10, inverse_estimators_100, inverse_estimators_10000],
               labels=['Batch Size 1', 'Batch Size 10', 'Batch Size 100', 'Batch Size 10000'],
               colors=['red', "yellow", "orange", 'green'],
               title='PSGD Error Vs. Batch size',
               fig_path="PSGD Error.png",
               r=r, m=m, X_matrix=X_matrix, y=Y)
