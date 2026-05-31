import numpy as np

from app.perceptron import PerceptronScratch


# test sprawdza poprawność nauki perceptrona
def test_perceptron_learns_simple_data():
    x = np.array(
        [
            [1, 1],
            [2, 2],
            [-1, -1],
            [-2, -2],
        ]
    )
    y = np.array([1, 1, -1, -1])

    perceptron = PerceptronScratch(lr=0.1, epochs=10)
    perceptron.fit(x, y)

    predictions = perceptron.predict(x)
    assert np.array_equal(predictions, y)
