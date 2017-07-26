from succinctly.datasets import get_dataset, linearly_separable as ls
import numpy as np
import cvxopt.solvers


def compute_w(multipliers, X, y):
    return np.sum(multipliers[i] * y[i] * X[i]
                  for i in range(len(y)))

def compute_b(w, X, y):
    return np.sum([y[i] - np.dot(w, X[i])
                   for i in range(len(X))])/len(X)


if __name__ == '__main__':
    X, y = get_dataset(ls.get_training_examples)
    m = X.shape[0]

    # Gram matrix - The matrix of all possible inner products of X.
    K = np.array([np.dot(X[i], X[j])
                  for j in range(m)
                  for i in range(m)]).reshape((m, m))

    P = cvxopt.matrix(np.outer(y, y) * K)
    q = cvxopt.matrix(-1 * np.ones(m))

    # Equality constraints
    A = cvxopt.matrix(y, (1, m))
    b = cvxopt.matrix(0.0)

    # Inequality constraints
    G = cvxopt.matrix(np.diag(-1 * np.ones(m)))
    h = cvxopt.matrix(np.zeros(m))

    # Solve the problem
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Lagrange multipliers
    multipliers = np.ravel(solution['x'])

    # Support vectors have positive multipliers.
    has_positive_multiplier = multipliers > 1e-7
    sv_multipliers = multipliers[has_positive_multiplier]

    support_vectors = X[has_positive_multiplier]
    support_vectors_y = y[has_positive_multiplier]


    w = compute_w(multipliers, X, y)
    w_from_sv = compute_w(sv_multipliers, support_vectors, support_vectors_y)

    # print(w)          # [0.44444446 1.11111114]
    print(w_from_sv)  # [0.44444453 1.11111128]

    b = compute_b(w, support_vectors, support_vectors_y)
    print(b) # -9.666668268506335