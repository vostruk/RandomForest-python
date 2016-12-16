import random

import numpy as np
import attr

from impurity_metrics import entropy, gini


class RandomForest:
    def __init__(self, num_trees, num_attributes, impurity_metric, max_tree_height=None):
        if impurity_metric not in ('gini', 'information gain', 'information gain ratio'):
            raise ValueError('impurity_metric attribute must be "gini" or "information gain" or "information gain ratio"')
        self.num_trees = num_trees
        self.num_attributes = num_attributes
        self.impurity_metric = impurity_metric
        self.max_tree_height = max_tree_height

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class DecisionTree:
    def __init__(self, max_attributes, criterion, categorical_features=None, max_tree_height=None):
        if criterion not in ('gini', 'information gain', 'information gain ratio'):
            raise ValueError('impurity_metric attribute must be "gini" or "information gain" or "information gain ratio"')
        self.max_attributes = max_attributes
        self.impurity_metric = gini if criterion == 'gini' else entropy
        self.max_tree_height = max_tree_height
        self.criterion = 'information gain ratio'
        self.categorical_features = [] if categorical_features is None else categorical_features

    def _build_node(self, X, y):
        unique_y = np.unique(y)
        if len(unique_y) == 1:
            return LeafNode(unique_y[0])

        all_attributes = X.shape[1]
        selected_attributes = random.sample(range(all_attributes), self.max_attributes)
        best_improved_impurity_value = np.inf
        for attribute in selected_attributes:
            for candidate_node in AttributeValueEqualNode.generate_candidate_nodes(X, y, attribute):
                equal_labels, not_equal_labels = candidate_node.split_labels()
                equal_impurity_value = self.impurity_metric(equal_labels)
                not_equal_impurity_value = self.impurity_metric(not_equal_labels)
                samples_num = len(y)

                improved_impurity_value = (len(equal_labels) * equal_impurity_value + len(not_equal_labels) * not_equal_impurity_value) / samples_num
                if self.criterion == 'information gain ratio':
                    intrinsic_value = entropy(X[:, attribute])
                    improved_impurity_value /= intrinsic_value

                if improved_impurity_value < best_improved_impurity_value:
                    best_node = candidate_node
                    best_improved_impurity_value = improved_impurity_value

        return best_node

    def _build_tree(self, X, y):
        root = self._build_node(X, y)
        nodes_stack = [root]
        while nodes_stack:
            node = nodes_stack.pop()
            non_leaf_children = [child for child in node.build_children(self) if not child.is_leaf()]
            nodes_stack.extend(non_leaf_children)
        return root


@attr.s
class AttributeValueEqualNode:
    attribute = attr.ib()
    value = attr.ib()
    X = attr.ib()
    y = attr.ib()
    equal_subtree = attr.ib(default=None)
    not_equal_subtree = attr.ib(default=None)

    @classmethod
    def generate_candidate_nodes(cls, X, y, attribute):
        values = np.unique(X[:, attribute])
        for value in values:
            yield cls(attribute, value, X, y)

    def split_labels(self):
        equal_indexes = self.X[:, self.attribute] == self.value
        equal_labels = self.y[equal_indexes]
        not_equal_labels = self.y[np.invert(equal_indexes)]
        return equal_labels, not_equal_labels

    def build_children(self, tree):
        equal_indexes = self.X[:, self.attribute] == self.value
        equal_X = self.X[equal_indexes]
        not_equal_X = self.X[np.invert(equal_indexes)]
        equal_y = self.y[equal_indexes]
        not_equal_y = self.y[np.invert(equal_indexes)]
        equal_node = tree._build_node(equal_X, equal_y)
        not_equal_node = tree._build_node(not_equal_X, not_equal_y)
        self.equal_subtree = equal_node
        self.not_equal_subtree = not_equal_node
        self.X = None
        self.y = None
        return equal_node, not_equal_node

    def is_leaf(self):
        return False


@attr.s
class LeafNode:
    class_ = attr.ib()

    def is_leaf(self):
        return True
