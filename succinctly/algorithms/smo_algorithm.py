import numpy as np
from random import randrange


# Written from the pseudo-code in :
# http://luthuli.cs.uiuc.edu/~daf/courses/optimization/Papers/smoTR.pdf
class SmoAlgorithm:
    def __init__(self, X, y, C, tol, kernel, use_linear_optim):
        self.X = X
        self.y = y
        self.m, self.n = np.shape(self.X)
        self.alphas = np.zeros(self.m)

        self.kernel = kernel
        self.C = C
        self.tol = tol

        self.errors = np.zeros(self.m)
        self.eps = 1e-3  # epsilon

        self.b = 0

        self.w = np.zeros(self.n)
        self.use_linear_optim = use_linear_optim

    # Compute the SVM output for example i
    # Note that Platt uses the convention w.x-b=0
    # while we have been using w.x+b in the book.
    def output(self, i):
        if self.use_linear_optim:
            # Equation 1
            return float(np.dot(self.w.T, self.X[i])) - self.b
        else:
            # Equation 10
            return np.sum([self.alphas[j] * self.y[j]
                           * self.kernel(self.X[j], self.X[i])
                           for j in range(self.m)]) - self.b

    # Try to solve the problem analytically.
    def take_step(self, i1, i2):

        if i1 == i2:
            return False

        a1 = self.alphas[i1]
        y1 = self.y[i1]
        X1 = self.X[i1]
        E1 = self.get_error(i1)

        s = y1 * self.y2

        # Compute the bounds of the new alpha2.
        if y1 != self.y2:
            # Equation 13
            L = max(0, self.a2 - a1)
            H = min(self.C, self.C + self.a2 - a1)
        else:
            # Equation 14
            L = max(0, self.a2 + a1 - self.C)
            H = min(self.C, self.a2 + a1)

        if L == H:
            return False

        k11 = self.kernel(X1, X1)
        k12 = self.kernel(X1, self.X[i2])
        k22 = self.kernel(self.X[i2], self.X[i2])

        # Compute the second derivative of the
        # objective function along the diagonal.
        # Equation 15
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            # Equation 16
            a2_new = self.a2 + self.y2 * (E1 - self.E2) / eta

            # Clip the new alpha so that is stays at the end of the line.
            # Equation 17
            if a2_new < L:
                a2_new = L
            elif a2_new > H:
                a2_new = H
        else:
            # Under unusual cicumstances, eta will not be positive.
            # Equation 19
            f1 = y1 * (E1 + self.b) - a1 * k11 - s * self.a2 * k12
            f2 = self.y2 * (self.E2 + self.b) - s * a1 * k12 \
                 - self.a2 * k22
            L1 = a1 + s(self.a2 - L)
            H1 = a1 + s * (self.a2 - H)
            Lobj = L1 * f1 + L * f2 + 0.5 * (L1 ** 2) * k11 \
                   + 0.5 * (L ** 2) * k22 + s * L * L1 * k12
            Hobj = H1 * f1 + H * f2 + 0.5 * (H1 ** 2) * k11 \
                   + 0.5 * (H ** 2) * k22 + s * H * H1 * k12

            if Lobj < Hobj - self.eps:
                a2_new = L
            elif Lobj > Hobj + self.eps:
                a2_new = H
            else:
                a2_new = self.a2

        # If alpha2 did not change enough the algorithm
        # returns without updating the multipliers.
        if abs(a2_new - self.a2) < self.eps * (a2_new + self.a2 \
                                                       + self.eps):
            return False

        # Equation 18
        a1_new = a1 + s * (self.a2 - a2_new)

        new_b = self.compute_b(E1, a1, a1_new, a2_new, k11, k12, k22, y1)

        delta_b = new_b - self.b

        self.b = new_b

        # Equation 22
        if self.use_linear_optim:
            self.w = self.w + y1 * (a1_new - a1) * X1 \
                     + self.y2 * (a2_new - self.a2) * self.X2

        # Update the error cache using the new Lagrange multipliers.
        delta1 = y1 * (a1_new - a1)
        delta2 = self.y2 * (a2_new - self.a2)

        # Update the error cache.
        for i in range(self.m):
            if 0 < self.alphas[i] < self.C:
                self.errors[i] += delta1 * self.kernel(X1, self.X[i]) + \
                                  delta2 * self.kernel(self.X2, self.X[i]) \
                                  - delta_b

        self.errors[i1] = 0
        self.errors[i2] = 0

        self.alphas[i1] = a1_new
        self.alphas[i2] = a2_new

        return True

    def compute_b(self, E1, a1, a1_new, a2_new, k11, k12, k22, y1):
        # Equation 20
        b1 = E1 + y1 * (a1_new - a1) * k11 + \
             self.y2 * (a2_new - self.a2) * k12 + self.b

        # Equation 21
        b2 = self.E2 + y1 * (a1_new - a1) * k12 + \
             self.y2 * (a2_new - self.a2) * k22 + self.b

        if (0 < a1_new) and (self.C > a1_new):
            new_b = b1
        elif (0 < a2_new) and (self.C > a2_new):
            new_b = b2
        else:
            new_b = (b1 + b2) / 2.0
        return new_b

    def get_error(self, i1):
        if 0 < self.alphas[i1] < self.C:
            return self.errors[i1]
        else:
            return self.output(i1) - self.y[i1]

    def second_heuristic(self, non_bound_indices):
        i1 = -1
        if len(non_bound_indices) > 1:
            max = 0

            for j in non_bound_indices:
                E1 = self.errors[j] - self.y[j]
                step = abs(E1 - self.E2)  # approximation
                if step > max:
                    max = step
                    i1 = j
        return i1

    def examine_example(self, i2):
        self.y2 = self.y[i2]
        self.a2 = self.alphas[i2]
        self.X2 = self.X[i2]
        self.E2 = self.get_error(i2)

        r2 = self.E2 * self.y2

        if not ((r2 < -self.tol and self.a2 < self.C) or
                    (r2 > self.tol and self.a2 > 0)):
            # The KKT conditions are met, SMO looks at another example.
            return 0

        # Second heuristic A: choose the Lagrange multiplier which
        # maximizes the absolute error.
        non_bound_idx = list(self.get_non_bound_indexes())
        i1 = self.second_heuristic(non_bound_idx)

        if i1 >= 0 and self.take_step(i1, i2):
            return 1

        # Second heuristic B: Look for examples making positive
        # progress by looping over all non-zero and non-C alpha,
        # starting at a random point.
        if len(non_bound_idx) > 0:
            rand_i = randrange(len(non_bound_idx))
            for i1 in non_bound_idx[rand_i:] + non_bound_idx[:rand_i]:
                if self.take_step(i1, i2):
                    return 1

        # Second heuristic C: Look for examples making positive progress
        # by looping over all possible examples, starting at a random
        # point.
        rand_i = randrange(self.m)
        all_indices = list(range(self.m))
        for i1 in all_indices[rand_i:] + all_indices[:rand_i]:
            if self.take_step(i1, i2):
                return 1

        # Extremely degenerate circumstances, SMO skips the first example.
        return 0

    def error(self, i2):
        return self.output(i2) - self.y2

    def get_non_bound_indexes(self):
        return np.where(np.logical_and(self.alphas > 0,
                                       self.alphas < self.C))[0]

    # First heuristic: loop  over examples where alpha is not 0 and not C
    # they are the most likely to violate the KKT conditions
    # (the non-bound subset).
    def first_heuristic(self):
        num_changed = 0
        non_bound_idx = self.get_non_bound_indexes()

        for i in non_bound_idx:
            num_changed += self.examine_example(i)
        return num_changed

    def main_routine(self):
        num_changed = 0
        examine_all = True

        while num_changed > 0 or examine_all:
            num_changed = 0

            if examine_all:
                for i in range(self.m):
                    num_changed += self.examine_example(i)
            else:
                num_changed += self.first_heuristic()

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

