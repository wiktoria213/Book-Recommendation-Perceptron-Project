import numpy as np


class PerceptronScratch:
    def __init__(self, lr=0.1, epochs=20):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        self.errors_per_epoch = []

    def fit(self, x, y):
        self.w = np.zeros(x.shape[1])
        self.b = 0.0
        self.errors_per_epoch = []

        # proces uczenie perceptronu
        for _ in range(self.epochs):
            errors = 0

            for xi, target in zip(x, y, strict=False):
                prediction = self.predict_one(xi)

                # aktualizacja wag przy błędnych perdykcjach
                if prediction != target:
                    self.w = self.w + self.lr * target * xi
                    self.b = self.b + self.lr * target
                    errors += 1

            self.errors_per_epoch.append(errors)
        return self

    def predict_one(self, x):
        # obliczanie wyniku dla pojedynczego rekordu
        score = np.dot(x, self.w) + self.b
        return 1 if score >= 0 else -1

    def decision_function(self, x):
        return np.dot(x, self.w) + self.b

    def predict(self, x):
        scores = self.decision_function(x)
        return np.where(scores >= 0, 1, -1)
