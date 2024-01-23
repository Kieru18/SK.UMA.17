import numpy as np
import matplotlib.pyplot as plt
import gc

from models.RegressionModel import RegressionModel


class CustomDecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, cost_function=None, optimizer=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.tree = None

    def fit(self, X, y, depth=0):
        # Check for stopping conditions
        if (self.max_depth is not None and depth == self.max_depth) or len(set(y)) == 1:
            return np.mean(y)

        if len(X) < self.min_samples_split:
            return np.mean(y)

        # Find the best split
        best_split = self.find_best_split(X, y)

        if best_split is None:
            return np.mean(y)

        feature_index, threshold, cost = best_split
        left_indices = X[:, feature_index] <= threshold
        right_indices = ~left_indices

        # Create a new node and recursively build subtrees
        node = {
            'feature_index': feature_index,
            'threshold': threshold,
            'left': self.fit(X[left_indices], y[left_indices], depth + 1),
            'right': self.fit(X[right_indices], y[right_indices], depth + 1)
        }

        return node

    def find_best_split(self, X, y):
        best_cost = float('inf')
        best_split = None

        for feature_index in range(X.shape[1]):
            unique_values = np.unique(X[:, feature_index])
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = ~left_indices

                if len(y[left_indices]) < self.min_samples_leaf or len(y[right_indices]) < self.min_samples_leaf:
                    continue

                cost = self.cost_function(y[left_indices], y[right_indices])

                if cost < best_cost:
                    best_cost = cost
                    best_split = (feature_index, threshold, cost)

        return best_split

    def predict_instance(self, x):
        if self.tree is None:
            raise ValueError("Tree not fitted.")

        node = self.tree
        while isinstance(node, dict):
            if x[node['feature_index']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']

        return node

    def predict(self, X):
        return np.array([self.predict_instance(x) for x in X])


class CustomRandomForestRegression:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, cost_function=None, optimizer=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.cost_function = cost_function
        self.optimizer = optimizer
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            # Draw a random subset of data with replacement
            idx = np.random.choice(len(X), len(X), replace=True)
            X_subset, y_subset = X[idx], y[idx]

            # Create a decision tree and fit it to the subset of data
            tree = CustomDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                cost_function=self.cost_function,
                optimizer=self.optimizer
            )
            tree.tree = tree.fit(X_subset, y_subset)

            # Add the trained tree to the forest
            self.trees.append(tree)

    def predict(self, X):
        # Make predictions for each tree in the forest
        predictions = np.array([tree.predict(X) for tree in self.trees])

        # Aggregate predictions using mean
        ensemble_prediction = np.mean(predictions, axis=0)
        return ensemble_prediction
