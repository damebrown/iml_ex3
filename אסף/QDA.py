# ------------------------------------------ imports -----------------------------------------------

import numpy as np
import operator


# ----------------------------------------------- Q7 -----------------------------------------------


class QDA:
    def __init__(self):
        return

    def find_N_y(self):
        unique, counts = np.unique(self.Y, return_counts=True)
        self.N_y = dict(zip(unique, counts))
        self.types_of_y = len(self.N_y)

    def find_pi_y(self):
        self.pi_y = {}
        for y, n_y in self.N_y.items():
            self.pi_y[y] = n_y / self.m

    def find_log_pi_y(self):
        self.log_pi_y = {}
        for y, n_y in self.N_y.items():
            self.log_pi_y[y] = np.log(self.pi_y[y])

    def find_mu_y(self):
        self.x_i_of_y_i = {}
        self.sum_x_i = {}
        self.mu_y = {}
        for y, n_y in self.N_y.items():
            self.x_i_of_y_i[y] = self.X[np.where(self.Y == y)]
            self.sum_x_i[y] = np.sum(self.x_i_of_y_i[y], axis=0)
            self.mu_y[y] = np.divide(self.sum_x_i[y], self.N_y[y])

    def find_Sigma_y(self):
        self.Sigma_y = {}
        for y, n_y in self.N_y.items():
            self.Sigma_y[y] = np.zeros((self.mu_y[y].shape[0], self.mu_y[y].shape[0]))
            for x in self.x_i_of_y_i[y]:
                subtract = np.subtract(x, self.mu_y[y])
                sub_t = np.reshape(subtract, (1, subtract.shape[0]))
                sub = np.reshape(subtract, (subtract.shape[0], 1))
                self.Sigma_y[y] = np.add(self.Sigma_y[y], np.dot(sub, sub_t))
            self.Sigma_y[y] = np.divide(self.Sigma_y[y], (self.N_y[y] - 1 ))
            # print("QDA - eigenvalues of sigma", y, ":",
            #       np.linalg.svd(self.Sigma_y[y], full_matrices=False, compute_uv=False))

    def find_Sigma_inv_y(self):
        self.Sigma_inv_y = {}
        for y, n_y in self.N_y.items():
            self.Sigma_inv_y[y] = np.linalg.inv(self.Sigma_y[y])

    def fit(self, X, Y):
        """
        stores the data
        """
        self.X = X
        self.Y = Y
        self.m = len(self.Y)
        QDA.find_N_y(self)
        QDA.find_pi_y(self)
        QDA.find_log_pi_y(self)
        QDA.find_mu_y(self)
        QDA.find_Sigma_y(self)
        QDA.find_Sigma_inv_y(self)

    def predict(self, x):
        delta_y = {}
        for y, n_y in self.N_y.items():
            delta_y_p1 = np.dot(np.transpose(x), self.Sigma_inv_y[y])
            delta_y_p1 = np.dot(delta_y_p1, self.mu_y[y])
            delta_y_p2 = np.dot(np.transpose(self.mu_y[y]), self.Sigma_inv_y[y])
            delta_y_p2 = np.dot(delta_y_p2, self.mu_y[y])
            delta_y_p2 = np.multiply(0.5, delta_y_p2)
            delta_y[y] = delta_y_p1 - delta_y_p2 + self.log_pi_y[y]

        return max(delta_y.items(), key=operator.itemgetter(1))[0]
