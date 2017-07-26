import numpy as np
from succinctly.datasets import get_dataset


def get_training_examples():
    X1 = np.array([[10,10],[8,6],[8,10],[8,8],[12,6],[9,5],[11,8],[11,5]])
    X2 = np.array([[10,13],[6,5],[6,9],[9,2],[14,8],[12,11],[10,13],[13,4]])

    y1 = np.ones(len(X1))
    y2 = np.ones(len(X2)) * -1
    return get_dataset(X1, y1, X2, y2)

