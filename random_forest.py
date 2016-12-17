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
        self.root = None

    def build_node(self, X, y):
        unique_y = np.unique(y)
        if len(unique_y) == 1:
            return LeafNode(unique_y[0])

        all_attributes = X.shape[1]
        selected_attributes = random.sample(range(all_attributes), self.max_attributes)
        best_improved_impurity_value = np.inf
        for attribute in selected_attributes:
            NodeType = AttributeValueEqualNode if attribute in self.categorical_features else AttributeValueLessNode
            for candidate_node in NodeType.generate_candidate_nodes(X, y, attribute):
                equal_labels, not_equal_labels = candidate_node.split_labels(X, y)
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

    def fit(self, X, y):
        self.root = self.build_node(X, y)
        nodes_stack = [(self.root, X, y)]
        while nodes_stack:
            node, node_X, node_y = nodes_stack.pop()
            non_leaf_children = [child for child in node.build_children(self, node_X, node_y) if not child[0].is_leaf()]
            nodes_stack.extend(non_leaf_children)

    def predict(self, X):
        predicted_labels_generator = (self.root.predict(x) for x in X)
        return np.fromiter(predicted_labels_generator, np.int)


@attr.s
class NonLeafNode:
    attribute = attr.ib()
    value = attr.ib()
    left_subtree = attr.ib(default=None)
    right_subtree = attr.ib(default=None)

    def split_labels(self, X, y):
        left_subtree_indexes = self._get_left_subtree_indexes(X)
        left_subtree_labels = y[left_subtree_indexes]
        right_subtree_labels = y[np.invert(left_subtree_indexes)]
        return left_subtree_labels, right_subtree_labels

    def build_children(self, tree, X, y):
        left_subtree_indexes = X[:, self.attribute] < self.value
        right_subtree_indexes = np.invert(left_subtree_indexes)
        left_subtree_X = X[left_subtree_indexes]
        right_subtree_X = X[right_subtree_indexes]
        left_subtree_y = y[left_subtree_indexes]
        right_subtree_y = y[right_subtree_indexes]
        left_subtree_node = tree.build_node(left_subtree_X, left_subtree_y)
        right_subtree_node = tree.build_node(right_subtree_X, right_subtree_y)
        self.left_subtree = left_subtree_node
        self.right_subtree = right_subtree_node
        return (
            (left_subtree_node, left_subtree_X, left_subtree_y),
            (right_subtree_node, right_subtree_X, right_subtree_y)
        )

    @staticmethod
    def is_leaf():
        return False


@attr.s
class AttributeValueLessNode(NonLeafNode):
    def _get_left_subtree_indexes(self, X):
        return X[:, self.attribute] < self.value

    def predict(self, x):
        if x[self.attribute] < self.value:
            return self.left_subtree.predict(x)
        return self.right_subtree.predict(x)

    @classmethod
    def generate_candidate_nodes(cls, X, y, attribute):
        values = np.unique(X[:, attribute])
        sorted_values = np.sort(values)
        for lower, higher in zip(sorted_values, sorted_values[1:]):
            split_point = (lower + higher) / 2.0
            yield cls(attribute, split_point, X, y)


@attr.s
class AttributeValueEqualNode(NonLeafNode):
    def _get_left_subtree_indexes(self, X):
        return X[:, self.attribute] == self.value

    def predict(self, x):
        if x[self.attribute] == self.value:
            return self.left_subtree.predict(x)
        return self.right_subtree.predict(x)

    @classmethod
    def generate_candidate_nodes(cls, X, y, attribute):
        values = np.unique(X[:, attribute])
        for value in values:
            yield cls(attribute, value, X, y)


@attr.s
class LeafNode:
    class_ = attr.ib()

    def predict(self, x):
        return self.class_

    @staticmethod
    def is_leaf():
        return True
