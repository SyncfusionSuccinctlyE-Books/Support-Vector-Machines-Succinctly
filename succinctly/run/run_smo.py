import numpy as np
from random import seed
from succinctly.datasets import linearly_separable, get_dataset
from succinctly.algorithms.smo_algorithm import SmoAlgorithm


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def compute_w(multipliers, X, y):
    return np.sum(multipliers[i] * y[i] * X[i] for i in range(len(y)))

if __name__ == '__main__':

    seed(5) # to have reproducible results

    X_data, y_data = get_dataset(linearly_separable.get_training_examples)

    smo = SmoAlgorithm(X_data, y_data, C=10, tol=0.001, kernel=linear_kernel, use_linear_optim=True)

    smo.main_routine()

    w = compute_w(smo.alphas, X_data, y_data)

    print('w = {}'.format(w))

    # -smo.b because Platt uses the convention w.x-b=0
    print('b = {}'.format(-smo.b))

    # w = [0.4443664  1.1105648]
    # b = -9.66268641132
