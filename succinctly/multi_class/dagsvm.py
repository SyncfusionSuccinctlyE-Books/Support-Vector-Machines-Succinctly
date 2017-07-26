from succinctly.multi_class import load_X, load_y
from itertools import combinations
from sklearn import svm
import numpy as np

def predict_class(X, classifiers, distinct_classes, class_pairs):
    results = []
    for x_row in X:

        class_list = list(distinct_classes)

        # After each prediction, delete the rejected class
        # until there is only one class
        while len(class_list) > 1:
            # We start with the pair of the first and last element in the list
            class_pair = (class_list[0], class_list[-1])
            classifier_index = class_pairs.index(class_pair)
            y_pred = classifiers[classifier_index].predict(x_row)

            if y_pred == 1:
                class_to_delete = class_pair[1]
            else:
                class_to_delete = class_pair[0]

            class_list.remove(class_to_delete)

        results.append(class_list[0])
    return np.array(results)


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
print(predict_class(X_to_predict, classifiers, set(y), class_pairs)) # prints [2 1]


