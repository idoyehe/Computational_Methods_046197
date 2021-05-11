import numpy as np
from lr_handlers import LrHandlerAdaGrad, LrHandlerDecreasing, LrHandlerConstant
from simulator import calc_variables
from estimators import estimate_a_analytic, estimate_a_constraint_GD, estimate_a_constraint_SGD
from plot_functions import plot_error, plot_estimation
np.random.seed(47)
from Wet_1.utils import *

n = 3
a0 = 0
a1 = 1
a2 = 0.5
a3 = -2
m = 10000
eps = 5e-4
r = 4
max_iter = 500
x, f, y = calc_variables(a0, a1, a2, a3, m)
X = np.asarray([[xi ** j for j in range(n + 1)] for xi in x])

a_estimated_analytic = estimate_a_analytic(X, y)

# plot_error(a_global_minima=a_estimated_analytic,
#            a_estimators= [a_vec_estimated_adagrad, a_vec_estimated_decrease],
#            labels= ['AdaGrad Error', 'Decreasing Learning Rate Error'],
#            title='Comparison between the error received with AdaGrad and Decreasing LR',
#            fig_path="Adagrad_decrease_error.png",
#            r=r, m=m, X=X, y=y)
# L = max(np.linalg.eig(np.transpose(X) @ X)[0]) / m
# learning_rates = [1/(10* L), 1/L, 10/L]
# estimators_vec = [estimate_a_constraint_GD(X=X, y=y, m=m, N=n, r=r,
#                                            lr_handler=LrHandlerConstant,
#                                            max_iter=max_iter, eps=eps, constant_lr=lr) for lr in learning_rates]
# plot_error(a_global_minima=a_estimated_analytic,
#            a_estimators=estimators_vec,
#            labels=[r'$\eta_t = \frac{1}{10L}$', r'$\eta_t = \frac{1}{L}$', r'$\eta_t = \frac{10}{L}$'],
#            title='The estimation error received with constant learning rates',
#            fig_path="estimation_error_of_constant_lr.png",
#            r=r, m=m, X=X, y=y)
# plot_error(a_global_minima=a_estimated_analytic,
#            a_estimators=[estimators_vec[1], a_vec_estimated_adagrad],
#            labels=[r'$\eta_t = \frac{1}{L}$', 'AdaGrad Error'],
#            title=r'Comparison between AdaGrad learning rate and $\eta_t = \frac{1}{L}$',
#            fig_path="estimation_error_comparison_constant_adagrad_lr.png",
#            r=r, m=m, X=X, y=y)
batch_sizes = [1, 10, 100, 10000]
sgd_batches_estimators_vec = [estimate_a_constraint_SGD(X=X, y=y, m=m, N=n, r=r, b=batch,
                                                        lr_handler=LrHandlerDecreasing,
                                                        max_iter=1000) for batch in batch_sizes]
plot_error(a_global_minima=a_estimated_analytic,
           a_estimators=[est[1] for est in sgd_batches_estimators_vec],
           labels=[r'batch size = ' + str(batch) for batch in batch_sizes],
           title=r'Comparison of SGD Error for different batch sizes and $\eta_t = \frac{D}{G\sqrt{t}}$',
           fig_path="comparison_of_SGD_errors_iterations.png",
           r=r, m=m, X=X, y=y)
# sgd_batches_estimators_vec_stop_time = [estimate_a_constraint_SGD(X=X, y=y, m=m, N=n, r=r, b=batch,
#                                                                   lr_handler=LrHandlerDecreasing,
#                                                                   max_iter=10000, stop_time=1)
#                                         for batch in batch_sizes]
# plot_error(a_global_minima=a_estimated_analytic,
#            a_estimators=[est[1] for est in sgd_batches_estimators_vec_stop_time],
#            x_axis_vec=[est[0] for est in sgd_batches_estimators_vec_stop_time],
#            xlim=[0, 0.2],
#            xlabel="running time [sec]",
#            labels=[r'batch size = ' + str(batch) for batch in batch_sizes],
#            title=r'Comparison of SGD Error for different batch sizes and $\eta_t = \frac{D}{G\sqrt{t}}$',
#            fig_path="comparison_of_SGD_errors_time.png",
#            r=r, m=m, X=X, y=y)





