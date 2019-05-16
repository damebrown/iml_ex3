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
            input_arr = np.array(x)
            data_arr = np.array(self._X[i])
            euclidean_distance = np.linalg.norm(input_arr - data_arr)
            deltas.append((self._X[i], euclidean_distance, self._y[i]))
        deltas.sort(key = lambda x: x[1])
        neighbors = deltas[:self._k]
        # monim = [neighbor[1] * neighbor[2] for neighbor in neighbors]
        monim = []
        mechane = 0
        for neighbor in neighbors:
            monim.append(neighbor[1] * neighbor[2])
            mechane += neighbor[1]
        if mechane == 0:
            hypothesis_set = [0]
        else:
            hypothesis_set = [mone / mechane for mone in monim]
        hypothesis = sum(hypothesis_set)
        return hypothesis
