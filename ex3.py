import sys

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


def h(x):
    w_0 = u_1 - u_0
    w_1 = 0.5 * ((u_0 * u_0) - (u_1 * u_1)) + np.math.log(pi_1 / pi_0)
    arr = []
    for i in x:
        arr.append(1 / (1 + np.math.exp(- (i * w_0) - w_1)))
    return arr


def q_5_b_ii():
    x0 = np.linspace(0, 2 * u_1)
    plt.plot(x0, h(x0))
    plt.show()


def f(x):
    return np.log(np.divide(x, 1 - x)) / 2 + 5


def q_5_b_iii():
    x = np.linspace(0, 1, 1000)
    # plot h(x) s.t. x~X|(Y=0)
    p_0 = norm.cdf(f(x), u_0, square_s)
    plt.plot(x, p_0)
    plt.title("cdf of h(x) while x~X|(Y=0)")
    plt.show()

    # plot h(x) s.t. x~X|(Y=1)
    p_1 = norm.cdf(f(x), u_1, square_s)
    plt.plot(x, p_1)
    plt.title("cdf of h(x) while x~X|(Y=1)")
    plt.show()


def q_5_b_iv():
    x = np.linspace(0, 1, 1000)
    # plot 1 - CDF(h(Z1)) s.t. Z1~X|(Y=0)
    p_0 = norm.cdf(f(x), u_0, square_s)
    line1, = plt.plot(x, 1 - p_0)
    label1 = "1 - CDF(h(Z1)) while Z1 ~ X|(Y=0)"

    # plot 1 - CDF(h(Z2)) s.t. Z2~X|(Y=1)
    p_1 = norm.cdf(f(x), u_1, square_s)
    line2, = plt.plot(x, 1 - p_1)
    label2 = "1 - CDF(h(Z2)) while Z2 ~ X|(Y=1)"

    plt.legend((line1, line2), (label1, label2))
    plt.show()


def q_5_b_vi():
    x0 = np.linspace(0, 2 * u_0)
    x1 = np.linspace(0, 2 * u_1)
    # plot pdf
    line1, = plt.plot(x0, norm.pdf(x0, u_0, square_s))
    label1 = 'pdf of 1st gaussian'
    line2, = plt.plot(x1, norm.pdf(x1, u_1))
    label2 = 'pdf of 2nd gaussian'
    points = [0.2, 0.4, 0.55, 0.95]
    min_a, min_b, min_c, min_d = f(points[0]), f(points[1]), f(points[2]), f(points[3])
    a = plt.axvline(x = min_a, linestyle = '--', color = 'red')
    b = plt.axvline(x = min_b, linestyle = '--', color = 'mediumseagreen')
    c = plt.axvline(x = min_c, linestyle = '--', color = 'magenta')
    d = plt.axvline(x = min_d, linestyle = '--', color = 'blue')
    plt.legend((line1, line2, a, b, c, d), (label1, label2,
                                            "min x for t = " + str(points[0]) + " is " + str(format(min_a, '.2f')),
                                            "min x for t = " + str(points[1]) + " is " + str(format(min_b, '.2f')),
                                            "min x for t = " + str(points[2]) + " is " + str(format(min_c, '.2f')),
                                            "min x for t = " + str(points[3]) + " is " + str(format(min_d, '.2f'))))
    plt.show()


def q_5_b_vii():
    x = np.linspace(0, 1, 1000)
    # plot the ROC
    p_0 = norm.cdf(f(x), u_0, square_s)
    p_1 = norm.cdf(f(x), u_1, square_s)
    plt.plot(1 - p_0, 1 - p_1)
    plt.title("ROC")
    plt.xlabel("TPR")
    plt.ylabel("FPR")

    plt.show()


def sqrt(x):
    arr = []
    for i in x:
        arr.append(np.math.sqrt(i))
    return arr


def log_plus_1(x):
    arr = []
    for i in x:
        arr.append(np.math.log(i) + 1)
    return arr


def q_6_b():
    x = np.linspace(0, 1, 1000)
    plt.plot(x, sqrt(x))
    plt.plot(x, x)
    plt.show()


def sbg(x):
    arr = []
    for i in x:
        if i <= 0.3:
            arr.append(2 * i)
        elif 0.3 <= i <= 0.5:
            arr.append(0.6)
        elif 0.5 <= i <= 0.9:
            arr.append(0.6 + (i - 0.5) / 2)
        elif 0.9 <= i <= 1:
            arr.append(0.8 + 2 * (i - 0.9))
    return arr


def q_6_c():
    x = np.linspace(0, 1, 1000)
    plt.plot(x, sbg(x))
    plt.show()


q_6_c()
