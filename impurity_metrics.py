import numpy as np


def entropy(x):
    _, counts = np.unique(x, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.dot(probabilities, np.log2(probabilities))


def gini(x):
    _, counts = np.unique(x, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.dot(probabilities, probabilities)
