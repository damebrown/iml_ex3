import numpy as np


class knn:
    _k_list = [1, 2, 5, 10, 100]

    def __init__(self, k):
        self._X = None
        self._y = None
        self._k = k

    def fit(self, X, y):
        """
        Simply stores the data.
        :param X: data
        :param y: labels
        """
        self._X = np.array(X).astype(float)
        self._y = np.array(y).astype(float)

    def predict(self, x):
        """
        Predicts x’s label according to the majority of its k nearest neighbors using Euclidean distance.
        """
        deltas = []
        for i in range(len(self._X)):
            input_arr = np.array(x).astype(float)
            data_arr = np.array(self._X[i]).astype(float)
            euclidean_distance = np.linalg.norm(input_arr - data_arr)
            deltas.append((data_arr, euclidean_distance, self._y[i]))
        deltas.sort(key = lambda x: x[1])
        neighbors = deltas[:self._k]
        monim = []
        mechane = 0
        for neighbor in neighbors:
            monim.append(neighbor[1] * neighbor[2])
            mechane += neighbor[1]
        if not mechane:
            hypothesis_set = [0]
        else:
            hypothesis_set = [mone / mechane for mone in monim]
        return sum(hypothesis_set)

    @staticmethod
    def get_ks():
        return knn._k_list
