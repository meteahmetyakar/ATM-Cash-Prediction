import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.root = None
        if random_state is not None:
            np.random.seed(random_state)

    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if (depth >= self.max_depth or num_samples < self.min_samples_split or
                num_samples <= self.min_samples_leaf):
            leaf_value = self._calculate_leaf_value(y)
            return self.Node(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y, num_features)
        if best_feature is None:
            leaf_value = self._calculate_leaf_value(y)
            return self.Node(value=leaf_value)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        if (sum(left_indices) < self.min_samples_leaf or sum(right_indices) < self.min_samples_leaf):
            leaf_value = self._calculate_leaf_value(y)
            return self.Node(value=leaf_value)

        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return self.Node(feature_index=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y, num_features):
        best_mse = float('inf')
        best_feature, best_threshold = None, None

        for feature_index in range(num_features):
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                left_indices = X_column <= threshold
                right_indices = X_column > threshold

                if sum(left_indices) < self.min_samples_leaf or sum(right_indices) < self.min_samples_leaf:
                    continue

                y_left, y_right = y[left_indices], y[right_indices]
                current_mse = self._calculate_mse(y_left, y_right)

                if current_mse < best_mse:
                    best_mse = current_mse
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_mse(self, y_left, y_right):
        mse_left = np.var(y_left) * len(y_left)
        mse_right = np.var(y_right) * len(y_right)
        total_mse = (mse_left + mse_right) / (len(y_left) + len(y_right))
        return total_mse

    def _calculate_leaf_value(self, y):
        return np.mean(y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

