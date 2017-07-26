from succinctly.multi_class import load_X, load_y
from sklearn.svm import LinearSVC
import numpy as np


X = load_X()
y = load_y()

clf = LinearSVC(C=1000, multi_class='crammer_singer')
clf.fit(X,y)

# Make predictions on two data points
X_to_predict = np.array([[5,5],[2,5]])
print(clf.predict(X_to_predict)) # prints [4 1]
