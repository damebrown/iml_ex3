# ------------------------------------------ imports -----------------------------------------------

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import qr
import matplotlib
import pandas as pd
from sklearn.linear_model import LogisticRegression

import seaborn as sns
import mpl_toolkits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from itertools import cycle


# ----------------------------------------------- Q7 -----------------------------------------------


def distance(instance1, instance2):
    """
    * Euclidean distance function i've found on web *
    """
    # just in case, if the instances are lists or tuples:
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    return np.linalg.norm(instance1 - instance2)

class knn:
    def __init__(self, k):
        """

        :param k: determines the number of nearest neighbors for the classifier
        """
        self.k = k

    def fit(self, X, Y):
        """
        stores the data
        :param X:
        :param y:
        """
        self.X = X
        self.Y = Y

    def predict(self, x):
        """
        method that predicts x’s label according to the majority of its k nearest neighbors (use Euclidean distance).
        :param x:
        :return:
        """
        distances = []
        for index in range(len(self.X)):
            dist = distance(x, self.X[index])
            distances.append((self.X[index], dist, self.Y[index]))
        distances.sort(key=lambda x: x[1])
        neighbors = distances[:self.k]
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
