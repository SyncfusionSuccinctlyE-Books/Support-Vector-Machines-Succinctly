import numpy as np

def perceptron_learning_algorithm(X, y):
    w = np.random.rand(3)   # can also be initialized at zero.
    misclassified_examples = predict(hypothesis, X, y, w)

    while misclassified_examples.any():
        x, expected_y = pick_one_from(misclassified_examples, X, y)
        w = w + x * expected_y  # update rule
        misclassified_examples = predict(hypothesis, X, y, w)

    return w

def hypothesis(x, w):
    return np.sign(np.dot(w, x))


# Make predictions on all data points
# and return the ones which are misclassified.
def predict(hypothesis_function, X, y, w):
    predictions = np.apply_along_axis(hypothesis_function, 1, X, w)
    misclassified = X[y != predictions]
    return misclassified

# Pick one misclassified example randomly
# and return it with its expected label.
def pick_one_from(misclassified_examples, X, y):
    np.random.shuffle(misclassified_examples)
    x = misclassified_examples[0]
    index = np.where(np.all(X == x, axis=1))
    return x, y[index]