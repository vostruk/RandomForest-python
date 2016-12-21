import random

import numpy as np
from scipy.stats import mode
import attr

from impurity_metrics import entropy, gini


class DecisionTree:
    def __init__(self, max_attributes, criterion, categorical_features=None, max_tree_height=None):
        if criterion not in ('gini', 'information gain', 'information gain ratio'):
            raise ValueError('impurity_metric attribute must be "gini" or "information gain" or "information gain ratio"')
        self.max_attributes = max_attributes
        self.impurity_metric = gini if criterion == 'gini' else entropy
        self.max_tree_height = max_tree_height
        self.criterion = criterion
        self.categorical_features = [] if categorical_features is None else categorical_features
        self.root = None

    def build_node(self, X, y, attributes_available, depth):
        new_constant_features = frozenset(_detect_constant_features(X, attributes_available))
        attributes_available = attributes_available - new_constant_features
        if len(attributes_available) == 0 or depth == self.max_tree_height:
            y_mode, _ = mode(y)
            return LeafNode(y_mode[0])
        unique_y = np.unique(y)
        if len(unique_y) == 1:  # uniform leaf
            return LeafNode(unique_y[0])

        num_attributes_to_be_selected = min(len(attributes_available), self.max_attributes)
        selected_attributes = random.sample(attributes_available, num_attributes_to_be_selected)
        best_score = -np.inf
        for attribute in selected_attributes:
            NodeType = AttributeValueEqualNode if attribute in self.categorical_features else AttributeValueLessNode
            for candidate_node in NodeType.generate_candidate_nodes(X, y, attribute, attributes_available, depth):
                score = self._score_node(candidate_node, attribute, X, y)
                if score > best_score:
                    best_node = candidate_node
                    best_score = score
        return best_node

    def _score_node(self, node, attribute, X, y):
        presplit_impurity = self.impurity_metric(y)
        left_labels, right_labels = node.split_labels(X, y)
        left_impurity = self.impurity_metric(left_labels)
        right_impurity = self.impurity_metric(right_labels)

        post_split_impurity = (len(left_labels) * left_impurity + len(right_labels) * right_impurity)
        gain = presplit_impurity - post_split_impurity
        if self.criterion == 'information gain ratio':
            intrinsic_value = entropy(X[:, attribute])
            gain /= intrinsic_value
        return gain

    def fit(self, X, y):
        all_attributes = X.shape[1]
        self.root = self.build_node(X, y, frozenset(range(all_attributes)), depth=0)
        nodes_stack = [(self.root, X, y)]
        while nodes_stack:
            node, node_X, node_y = nodes_stack.pop()
            non_leaf_children = [child for child in node.build_children(self, node_X, node_y) if not child[0].is_leaf()]
            nodes_stack.extend(non_leaf_children)

    def predict(self, X):
        if self.root is None:
            raise ValueError('You must first train the tree using fit before predicting data')
        predicted_labels_generator = (self.root.predict(x) for x in X)
        return np.fromiter(predicted_labels_generator, np.int, count=len(X))


@attr.s
class NonLeafNode:
    attribute = attr.ib()
    value = attr.ib()
    attributes_available = attr.ib()
    depth = attr.ib()
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
        left_subtree_node = tree.build_node(left_subtree_X, left_subtree_y, self.attributes_available, self.depth + 1)
        right_subtree_node = tree.build_node(right_subtree_X, right_subtree_y, self.attributes_available, self.depth + 1)
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
    def generate_candidate_nodes(cls, X, y, attribute, attributes_available, depth):
        values = np.unique(X[:, attribute])
        sorted_values = np.sort(values)
        for lower, higher in zip(sorted_values, sorted_values[1:]):
            split_point = (lower + higher) / 2.0
            yield cls(attribute, split_point, attributes_available, depth)


@attr.s
class AttributeValueEqualNode(NonLeafNode):
    def _get_left_subtree_indexes(self, X):
        return X[:, self.attribute] == self.value

    def predict(self, x):
        if x[self.attribute] == self.value:
            return self.left_subtree.predict(x)
        return self.right_subtree.predict(x)

    @classmethod
    def generate_candidate_nodes(cls, X, y, attribute, attributes_available, depth):
        values = np.unique(X[:, attribute])
        for value in values:
            yield cls(attribute, value, attributes_available, depth)


@attr.s
class LeafNode:
    class_ = attr.ib()

    def predict(self, x):
        return self.class_

    @staticmethod
    def is_leaf():
        return True


def _detect_constant_features(a, features_to_check):
    for feature in features_to_check:
        if len(np.unique(a[:, feature])) == 1:
            yield feature