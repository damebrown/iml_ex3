import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from numpy import random as rnd

import matplotlib.pyplot as plt
import numpy as np

ITERATIONS = 10

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


def sort_by_index(index, list):
    arr = []
    for i in index:
        arr.append(list[i])
    return arr


def q_7_ab():
    test_size = 1000
    data = pd.read_csv('seperatedData.csv', sep = ',', header = None)
    data = data.drop(58, axis = 'columns')
    data = data.drop(0, axis = 'rows')
    data = data.to_numpy()
    tpr, fpr = [0], [0]
    total_tpr, total_fpr = [[] for i in range(ITERATIONS)], [[] for i in range(ITERATIONS)]
    # total_tpr, total_fpr = [], []
    avg_tpr, avg_fpr = [], []
    for i in range(ITERATIONS):
        regression(data, test_size, tpr, fpr, total_tpr, total_fpr, i)
        # total_tpr.append(a[0]), total_fpr.append(a[1])
        tpr, fpr = [0], [0]
    total_tpr, total_fpr = np.array(total_tpr), np.array(total_fpr)
    m = len(total_tpr[0])
    for _ in range(ITERATIONS):
        if len(total_tpr[_]) < m:
            m = len(total_tpr[_])
    for i in range(m):
        tpr_s, fpr_s = 0, 0
        for _ in range(ITERATIONS):
            tpr_s += total_tpr[_][i]
            fpr_s += total_fpr[_][i]
        avg_tpr.append(tpr_s / ITERATIONS)
        avg_fpr.append(fpr_s / ITERATIONS)
    plt.plot(avg_fpr, avg_tpr)
    plt.show()


def regression(data, test_size, tpr, fpr, total_tpr, total_fpr, i):
    # test and train sets initialization
    indices = np.random.choice(len(data), len(data), replace = False)
    test_set = data[indices[:test_size]]
    test_y = test_set[:, 57]
    train_set = data[indices[test_size:]]
    train_y = train_set[:, 57]
    # modelling
    model = LogisticRegression()
    model.fit(train_set, train_y)
    probabilities = model.predict_proba(test_set)[:, 1]
    # sorted_y = test_y[probabilities[:, 1].argsort()]  # sort array with regards to 1th column
    sorted_y = np.array(sort_by_index(np.argsort(-probabilities), test_y)).astype(np.int)
    cum_y = np.cumsum(sorted_y)
    _np = sum(sorted_y)
    _nn = len(sorted_y) - _np
    n_i = 1
    # for j in range(1, len(n_i)):
    for j in range(1, _np):
        for k in range(n_i, len(cum_y)):
            if cum_y[k - 1] == j:
                n_i = k
                tpr.append(j / _np)
                fpr.append((n_i - j + 1) / _nn)
                # tpr.append((n_i[j - 1] / _np))
                # fpr.append(((j - n_i[j - 1]) / _nn))
                break
    fpr.append(1), tpr.append(1)
    total_tpr[i] = tpr
    total_fpr[i] = fpr
    # plt.plot(fpr, tpr)


q_7_ab()
