from random import seed

import numpy as np
import time
from succinctly.datasets import linearly_separable, get_dataset
from succinctly.algorithms.smo_algorithm import SmoAlgorithm
from sklearn import svm

if __name__ == '__main__':

    def linear_kernel(x1, x2):
        return np.dot(x1, x2)

    def compute_w(multipliers, X, y):
        return np.sum(multipliers[i] * y[i] * X[i] for i in range(len(y)))

    np.random.seed(5)
    seed(5)

    X_data, y_data = get_dataset(linearly_separable.get_training_examples)


    smo = SmoAlgorithm(X_data, y_data, C=10, tol=0.001, kernel=linear_kernel, use_linear_optim=True)
    start_time = time.time()
    smo.main_routine()
    print("SmoAlgorithm took %s seconds ---" % (time.time() - start_time))

    w = compute_w(smo.alphas, X_data, y_data)

    print('w = {}'.format(w))
    print('b = {}'.format(-smo.b))  # -smo.b because Platt uses the convention w.x-b=0

    # Train a linear SVM using sklearn to check our result is the same (or close)

    clf = svm.SVC(kernel='linear', C=10, tol=0.001)
    start_time = time.time()
    clf.fit(X_data, y_data)

    print("SMO from sklearn took %s seconds ---" % (time.time() - start_time))

    w = clf.coef_[0]
    b = clf.intercept_[0]

    print('w = {}'.format(w))
    print('b = {}'.format(b))

