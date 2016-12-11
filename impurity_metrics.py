import numpy as np


def entropy(x):
    _, counts = np.unique(x, return_counts=True)
    probabilities = counts / counts.sum()
    return -(probabilities * np.log2(probabilities)).sum()


def gini(x):
    _, counts = np.unique(x, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.square(probabilities).sum()
