import numpy as np
from succinctly.datasets import get_dataset


def get_training_examples():
    X1 = np.array([[10,10],[6,6],[6,11],[3,15],[12,6],[9,5],[16,3],[11,5]])
    X2 = np.array([[3,6],[6,3],[2,9],[9,2],[18,1],[1,18],[1,13],[13,1]])

    y1 = np.ones(len(X1))
    y2 = np.ones(len(X2)) * -1
    return get_dataset(X1, y1, X2, y2)
