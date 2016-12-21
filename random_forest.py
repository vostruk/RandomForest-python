import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin

from decision_tree import DecisionTree


class RandomForest(ClassifierMixin, BaseEstimator):
    def __init__(self, num_trees=100, num_attributes=None, impurity_metric='gini', categorical_features=None, max_tree_height=None):
        impurity_metric = impurity_metric or 'gini'
        if impurity_metric not in ('gini', 'information gain', 'information gain ratio'):
            raise ValueError('impurity_metric attribute must be "gini" or "information gain" or "information gain ratio"')
        self.num_trees = num_trees
        self.num_attributes = num_attributes
        self.criterion = impurity_metric
        self.categorical_features = categorical_features
        self.max_tree_height = max_tree_height
        self.forest = None

    def fit(self, X, y):
        n = len(y)
        self.forest = [
            DecisionTree(self.num_attributes, self.criterion, self.categorical_features, self.max_tree_height)
            for _ in range(self.num_trees)
        ]
        for tree in self.forest:
            sampled_indices = np.random.choice(n, n, replace=True)
            sampled_X = X[sampled_indices]
            sampled_y = y[sampled_indices]
            tree.fit(sampled_X, sampled_y)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.forest])
        result, _ = mode(predictions)
        return result[0]
