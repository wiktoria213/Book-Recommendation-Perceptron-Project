import numpy as np


class PerceptronScratch:
    def __init__(self, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.errors_per_epoch = []

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.errors_per_epoch = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                prediction = self.predict_one(xi)
                if prediction != target:
                    self.w = self.w + self.lr * target * xi
                    self.b = self.b + self.lr * target
                    errors += 1

            self.errors_per_epoch.append(errors)
        return self

    def predict_one(self, x):
        score = np.dot(x, self.w) + self.b
        return 1 if score >= 0 else -1

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)