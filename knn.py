import numpy as np


class knn:
    def __init__(self, k):
        self._X = None
        self._y = None
        self._k = k
        self._k_list = [1, 2, 5, 10, 100]

    def fit(self, X, y):
        """
        stores the data
        :param X: data
        :param y: labels
        """
        self._X = X
        self._y = y

    def predict(self, x):
        """
        method that predicts x’s label according to the majority of its k nearest neighbors using Euclidean distance.
        :param x:
        :return:
        """
        deltas = []
        for i in range(len(self._X)):
            dist = distance(x, self._X[i])
            deltas.append((self._X[i], dist, self._y[i]))
        deltas.sort(key = lambda x: x[1])
        neighbors = deltas[:self._k]
        numerators = [neighbor[1] * neighbor[2] for neighbor in neighbors]
        denominator = 0
        for neighbor in neighbors:
            denominator += neighbor[1]
        if denominator == 0:
            h_s_list = [0]
        else:
            h_s_list = [Numerator / denominator for Numerator in numerators]
        h_s = sum(h_s_list)
        return h_s


def distance(instance1, instance2):
    """
    * Euclidean distance function i've found on web *
    """
    # just in case, if the instances are lists or tuples:
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    return np.linalg.norm(instance1 - instance2)
