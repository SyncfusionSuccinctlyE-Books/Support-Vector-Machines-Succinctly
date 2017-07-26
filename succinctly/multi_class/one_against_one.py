from succinctly.multi_class import load_X, load_y
from scipy.stats import mode
from itertools import combinations
from sklearn import svm
import numpy as np


# Predict the class having the max number of votes
def predict_class(X, classifiers, class_pairs):
    predictions = np.zeros((X.shape[0], len(classifiers)))
    for idx, clf in enumerate(classifiers):
        class_pair = class_pairs[idx]
        prediction = clf.predict(X)
        predictions[:, idx] = np.where(prediction == 1, class_pair[0], class_pair[1])
    return mode(predictions, axis=1)[0].ravel().astype(int)

X = load_X()
y = load_y()

# Create datasets
training_data = []
class_pairs = list(combinations(set(y), 2))
for class_pair in class_pairs:
    class_mask = np.where((y == class_pair[0]) | (y == class_pair[1]))
    y_i = np.where(y[class_mask] == class_pair[0], 1, -1)
    training_data.append((X[class_mask], y_i))

# Train one classifier per class
classifiers = []
for data in training_data:
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(data[0], data[1])
    classifiers.append(clf)

# Make predictions on two data points
X_to_predict = np.array([[5,5],[2,5]])
print(predict_class(X_to_predict, classifiers, class_pairs)) # prints [2 1]


