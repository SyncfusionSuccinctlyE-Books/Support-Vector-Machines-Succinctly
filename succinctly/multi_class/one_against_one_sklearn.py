from succinctly.multi_class import load_X, load_y
from sklearn import svm
import numpy as np

X = load_X()
y = load_y()

# Train a multi-class classifier
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X,y)

# Make predictions on two data points
X_to_predict = np.array([[5,5],[2,5]])
print(clf.predict(X_to_predict))  # return [2, 1]
