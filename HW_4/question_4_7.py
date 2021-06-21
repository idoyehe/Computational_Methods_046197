import numpy as np

a = np.sqrt(2)


def _stop_condition(current_delta):
    global a
    return current_delta <= (1e-50) / a


def _next_delta_1(current_delta):
    return (current_delta ** 2) / (2 * (1 + current_delta))


def _next_delta_2(current_delta):
    return current_delta / 2


def _next_delta_3(current_delta):
    return (current_delta ** 2) / 2


def _evaluations(update_function, delta_0):
    k = 0
    current_delta = delta_0
    while not _stop_condition(current_delta):
        k += 1
        current_delta = update_function(current_delta)
    return k


if __name__ == '__main__':
    delta_0 = a - 1
    print("1st expression reached bound in {} iterations".format(_evaluations(_next_delta_1, delta_0)))
    print("2nd expression reached bound in {} iterations".format(_evaluations(_next_delta_2, delta_0)))
    print("3rd expression reached bound in {} iterations".format(_evaluations(_next_delta_3, delta_0)))
