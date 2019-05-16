import numpy as np
import operator


class LDA:
    def __init__(self):
        self.n_y = None
        self.y_types = None
        self.s = None
        self.X = None
        self.Y = None
        self.m = None
        self.pi_y = {}
        self.log_of_pi_y = {}
        self.x_i = {}
        self.sum_x_i = {}
        self.u_y = {}
        self.s_y = {}
        self.s_inverse = None

    def get_n_y(self):
        unique, counts = np.unique(self.Y, True)
        self.n_y = dict(zip(unique, counts))
        self.y_types = len(self.n_y)

    def get_pi_y(self):
        for y, y_numbers in self.n_y.items():
            self.pi_y[y] = y_numbers / self.m

    def get_log_pi_y(self):
        for y, y_numbers in self.n_y.items():
            self.log_of_pi_y[y] = np.log(self.pi_y[y])

    def get_mu_y(self):
        for y, y_numbers in self.n_y.items():
            self.x_i[y] = self.X[np.where(self.Y == y)]
            self.sum_x_i[y] = np.sum(self.x_i[y], axis = 0)
            self.u_y[y] = np.divide(self.sum_x_i[y], self.n_y[y])

    def get_s_y(self):
        self.s_y = {}
        for y, y_numbers in self.n_y.items():
            self.s_y[y] = np.zeros((self.u_y[y].shape[0], self.u_y[y].shape[0]))
            for x in self.x_i[y]:
                x_i_sub_mu_y = np.subtract(x, self.u_y[y])
                sub_t = np.reshape(x_i_sub_mu_y, (1, x_i_sub_mu_y.shape[0]))
                sub = np.reshape(x_i_sub_mu_y, (x_i_sub_mu_y.shape[0], 1))
                self.s_y[y] = np.add(self.s_y[y], np.dot(sub, sub_t))
        self.s = np.add(self.s_y[0], self.s_y[1])
        self.s = np.divide(self.s, self.m - 2)
        self.s_inverse = np.linalg.inv(self.s)

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.m = len(self.Y)
        LDA.get_n_y(self)
        LDA.get_pi_y(self)
        LDA.get_log_pi_y(self)
        LDA.get_mu_y(self)
        LDA.get_s_y(self)

    def predict(self, x):
        delta_y = {}
        for y, y_numbers in self.n_y.items():
            y_p1 = np.dot(np.dot(x.t, self.s_inverse), self.u_y[y])
            delta_y_p2 = np.multiply(0.5, np.dot(np.dot(self.u_y[y].T, self.s_inverse), self.u_y[y]))
            delta_y[y] = y_p1 - delta_y_p2 + self.log_of_pi_y[y]
        return max(delta_y.items(), operator.itemgetter(1))[0]
