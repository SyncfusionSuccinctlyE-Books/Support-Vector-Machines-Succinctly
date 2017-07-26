import numpy as np

def get_dataset(get_examples):
    X1, y1, X2, y2 = get_examples()
    X, y = get_dataset_for(X1, y1, X2, y2)
    return X, y

def get_dataset_for(X1, y1, X2, y2):
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y

def get_generated_dataset(get_examples, n):
    X1, y1, X2, y2 = get_examples(n)
    X, y = get_dataset_for(X1, y1, X2, y2)
    return X, y