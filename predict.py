import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def custom_resample(indices, n_samples, replace=True, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    if replace:
        return np.random.choice(indices, size=n_samples, replace=True)
    else:
        return np.random.choice(indices, size=n_samples, replace=False)

class BootstrapOptimisticTreeConstruction:
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
        best_gain = float('-inf')
        best_feature, best_threshold = None, None

        for feature_index in range(num_features):
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                left_indices = X_column <= threshold
                right_indices = X_column > threshold

                if sum(left_indices) < self.min_samples_leaf or sum(right_indices) < self.min_samples_leaf:
                    continue

                gain = self._information_gain(y, y[left_indices], y[right_indices])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, y_left, y_right):
        parent_var = np.var(y) * len(y)
        left_var = np.var(y_left) * len(y_left)
        right_var = np.var(y_right) * len(y_right)
        gain = parent_var - (left_var + right_var)
        return gain

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

    def get_params(self):
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state
        }

class BaggingRegressor:
    def __init__(self, base_estimator, n_estimators=10, max_samples=1.0, max_features=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self.estimators = []
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.estimators = []

        for _ in range(self.n_estimators):
            # Bootstrap sampling for rows
            n_sample_subset = int(self.max_samples * n_samples)
            row_indices = custom_resample(np.arange(n_samples), n_samples=n_sample_subset, replace=True, random_state=self.random_state)

            # Random feature sampling
            n_feature_subset = int(self.max_features * n_features)
            feature_indices = np.random.choice(np.arange(n_features), size=n_feature_subset, replace=False)

            # Subset of the data
            X_subset = X[row_indices][:, feature_indices]
            y_subset = y[row_indices]

            # Clone and train the base estimator
            estimator = self._clone_estimator(self.base_estimator)
            estimator.fit(X_subset, y_subset)

            # Save the estimator and its feature indices
            self.estimators.append((estimator, feature_indices))

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.estimators)))

        for i, (estimator, feature_indices) in enumerate(self.estimators):
            predictions[:, i] = estimator.predict(X[:, feature_indices])

        return np.mean(predictions, axis=1)

    def _clone_estimator(self, estimator):
        # Clone the base estimator by creating a new instance with the same parameters
        return type(estimator)(**estimator.get_params())

    def set_params(self, **params):
        """
        Set parameters for the bagging regressor dynamically.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def get_params(self, deep=True):
        """
        Get parameters of the bagging regressor as a dictionary.
        """
        return {
            'base_estimator': self.base_estimator,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'max_features': self.max_features,
            'random_state': self.random_state
        }

# Data preprocessing and model evaluation
if __name__ == "__main__":
    # Load dataset
    file_path = "updated_ATM.csv"  # Replace with the correct file path
    data = pd.read_csv(file_path)

    # Display first 5 rows
    print("\nData Preview:\n", data.head())

    # Data structure
    print("\nData Info:\n")
    data.info()

    # Check for missing values
    print("\nMissing Values:\n", data.isnull().sum())

    # Features and target variable
    X = data.drop(columns=["total_amount_withdrawn", "transaction_date", "atm_name"], axis=1)
    Y = data["total_amount_withdrawn"]

    # Transform target variable
    pt = PowerTransformer()
    Y = pt.fit_transform(Y.values.reshape(-1, 1)).flatten()

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Base estimator
    base_estimator = BootstrapOptimisticTreeConstruction(max_depth=5, random_state=42)

    # Bagging Regressor
    bagging_regressor = BaggingRegressor(base_estimator=base_estimator, n_estimators=10, max_samples=0.8, max_features=0.8, random_state=42)

    # Fit and predict
    bagging_regressor.fit(X_train, y_train)
    y_pred = bagging_regressor.predict(X_test)

    # Evaluate model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nModel Performance Metrics:\n")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R^2: {r2:.2f}")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Actual", alpha=0.8, color="blue")
    plt.plot(y_pred, label="Predicted", alpha=0.8, color="orange")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot error distribution
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=30, color="red")
    plt.title("Error Distribution")
    plt.tight_layout()
    plt.show()
