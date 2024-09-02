import numpy as np
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        y_pred = [self._predict(i) for i in x]
        return y_pred

    def _predict(self, x):
        # compute the distance between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.x_train]

        # get closest k samples, labels
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

