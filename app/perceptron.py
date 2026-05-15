import numpy as np

class PerceptronScratch:
    def __init__(self, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.errors_per_epoch = []

    def fit(self, X, y):
        #
        return self

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)