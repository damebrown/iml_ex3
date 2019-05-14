from scipy.stats import norm
from numpy import random as rnd
import matplotlib.pyplot as plt
import numpy as np

u_0 = 4
u_1 = 6
square_s = 1
pi_0, pi_1 = 1 / 2, 1 / 2
# g_0 = rnd.normal(u_0, square_s)
# g_1 = rnd.normal(u_1, square_s)
g_00 = norm.cdf(x = square_s, loc = u_0)
g_01 = norm.cdf(x = square_s, loc = u_1)


def q_5_b_i():
    x0 = np.linspace(0, 2 * u_0)
    x1 = np.linspace(0, 2 * u_1)
    # plot pdf
    line1, = plt.plot(x0, norm.pdf(x0, u_0, square_s))
    label1 = 'pdf of 1st gaussian'
    line2, = plt.plot(x1, norm.pdf(x1, u_1))
    label2 = 'pdf of 2nd gaussian'
    # plot cdf
    line3, = plt.plot(x0, norm.cdf(x0, u_0, square_s))
    label3 = 'cdf of 1st gaussian'
    line4, = plt.plot(x1, norm.cdf(x1, u_1))
    label4 = 'cdf of 2nd gaussian'
    plt.legend((line1, line2, line3, line4), (label1, label2, label3, label4))
    plt.show()


q_5_b_i()
