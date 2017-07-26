import numpy as np

# This is the dataset used in most of the book.
# Usage:
#
# from succinctly.datasets import linearly_separable, get_dataset
#
# # Load training examples
# X_train, y_train = get_dataset(linearly_separable.get_training_examples)
#
# # Load test examples
# X_test, y_test = get_dataset(linearly_separable.get_test_examples)

def get_training_examples():
    X1 = np.array([[8, 7], [4, 10], [9, 7], [7, 10],
                   [9, 6], [4, 8], [10, 10]])
    y1 = np.ones(len(X1))
    X2 = np.array([[2, 7], [8, 3], [7, 5], [4, 4],
                   [4, 6], [1, 3], [2, 5]])
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

def get_test_examples():
    X1 = np.array([[2, 9], [1, 10], [1, 11], [3, 9], [11, 5],
                   [10, 6], [10, 11], [7, 8], [8, 8], [4, 11],
                   [9, 9], [7, 7], [11, 7], [5, 8], [6, 10]])
    X2 = np.array([[11, 2], [11, 3], [1, 7], [5, 5], [6, 4],
                   [9, 4], [2, 6], [9, 3], [7, 4], [7, 2], [4, 5],
                   [3, 6], [1, 6], [2, 3], [1, 1], [4, 2], [4, 3]])
    y1 = np.ones(len(X1))
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2
