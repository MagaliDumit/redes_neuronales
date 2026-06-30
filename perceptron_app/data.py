import numpy as np


class DataHandler:
    @staticmethod
    def train_test_split(X, y, test_size=0.2, shuffle=True):
        if shuffle:
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]
        split_idx = int(X.shape[0] * (1 - test_size))
        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    @staticmethod
    def load_tp1_data(data_path, target_path):
        X = np.loadtxt(data_path)
        y = np.loadtxt(target_path).reshape(-1, 1)
        return X, y

    @staticmethod
    def normalize_targets(y, low=-0.9, high=0.9):
        y_min = y.min()
        y_max = y.max()
        y_norm = low + (y - y_min) * (high - low) / (y_max - y_min)
        return y_norm, y_min, y_max

    @staticmethod
    def denormalize_targets(y_norm, y_min, y_max, low=-0.9, high=0.9):
        return y_min + (y_norm - low) * (y_max - y_min) / (high - low)

    @staticmethod
    def evaluate_mse(model, X_test, y_test):
        errors = 0
        for V, target in zip(X_test, y_test):
            pred, _, _ = model.forward(V)
            errors += (target - pred) ** 2
        return float(errors / len(X_test))

    @staticmethod
    def evaluate_accuracy(model, X_test, y_test):
        correct = 0
        for V, target in zip(X_test, y_test):
            pred, _, _ = model.forward(V)
            if np.sign(pred) == np.sign(target):
                correct += 1
        return correct / len(X_test)
