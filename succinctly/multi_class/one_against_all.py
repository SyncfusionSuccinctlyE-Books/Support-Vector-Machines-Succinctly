from succinctly.multi_class import load_X, load_y
import numpy as np
from sklearn import svm

def predict_class(X, classifiers):
    predictions = np.zeros((X.shape[0], len(classifiers)))
    for idx, clf in enumerate(classifiers):
        predictions[:, idx] = clf.predict(X)

    # return the class number if only one classifier predicted it
    # return zero otherwise
    return np.where((predictions == 1).sum(1) == 1,
                    (predictions == 1).argmax(axis=1) + 1,
                    0)

# Load the dataset
X = load_X()
y = load_y()

# Transform the 4 classes y in 4 binary classes y
y_1 = np.where(y == 1, 1, -1)
y_2 = np.where(y == 2, 1, -1)
y_3 = np.where(y == 3, 1, -1)
y_4 = np.where(y == 4, 1, -1)

y_list = [y_1, y_2, y_3, y_4]

# Train one binary classifier on each problem
classifiers = []
for y_i in y_list:
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y_i)
    classifiers.append(clf)

# Make predictions on two data points
X_to_predict = np.array([[5,5],[2,5]])
print(predict_class(X_to_predict, classifiers)) # prints [0 1]



