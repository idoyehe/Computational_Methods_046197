import numpy as np


def calc_f(xi, a):
    return sum([a[j]*xi**j for j in range(len(a))])


def calc_variables(a0, a1, a2, a3, m):
    x = []
    f = []
    y = []
    for i in range(int(m)):
        xi = 2*np.random.rand() - 1
        x.append(xi)
        f.append(calc_f(xi=xi, a=[a0, a1, a2, a3]))
        y.append(f[-1] + np.random.randn()/np.sqrt(2))
    return x, f, y