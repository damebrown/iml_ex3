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
from sklearn.metrics import log_loss
from itertools import cycle
from scipy.special import logit
import operator

# from knn import knn
# from QDA import QDA
# from LDA import LDA

# ------------------------------------------ constants ---------------------------------------------

# q5
mu_0 = 4
mu_1 = 6
w = np.array([2, -10])

# q7

PROCEDURE_REPETITIONS = 10
df = pd.read_csv('seperatedData.csv', header=None)
df = df.drop(58, axis = 'columns')
df = df.drop(0, axis = 'rows')
x_slice = df.iloc[:, :-1].values
y_slice = df.iloc[:, 57].values

# q7_d

DIM = [1, 4, 21, 22, 32]


# ----------------------------------------------- Q5 -----------------------------------------------

def q5_i():
    x_0 = np.linspace(norm.ppf(0.01, loc=mu_0), norm.ppf(0.99, loc=mu_0), 100)
    x_1 = np.linspace(norm.ppf(0.01, loc=mu_1), norm.ppf(0.99, loc=mu_1), 100)

    plt.plot(x_0, norm.pdf(x_0, loc=mu_0), label='$\mu$0 = 4')
    plt.plot(x_1, norm.pdf(x_1, loc=mu_1), label='$\mu$1 = 6')

    plt.legend(loc='best')
    plt.title("pdf")
    plt.show()

    plt.plot(x_0, norm.cdf(x_0, loc=mu_0), label='$\mu$0 = 4')
    plt.plot(x_1, norm.cdf(x_1, loc=mu_1), label='$\mu$1 = 6')

    plt.legend(loc='best')
    plt.title("cdf")
    plt.show()


def logit_inv(x):
    res = np.zeros(len(x))
    for i in range(len(x)):
        x_w = w[0] * x[i]
        b = w[1]
        cur_res = (np.exp(x_w + b)) / (1 + np.exp(x_w + b))
        res[i] = cur_res
    return res


def q5_ii():
    x = np.linspace(0, 10, 1000)
    plt.plot(x, logit_inv(x))
    plt.title("h(x) as a function of x")
    plt.show()


def q5_iii():
    x = np.linspace(0, 1, 1000)

    plt.plot(x, norm.cdf(logit(x) / 2 + 5, mu_0))
    plt.title("cdf distribution of h(x) for x ~ X|(Y = 0)")
    plt.show()

    plt.plot(x, norm.cdf(logit(x) / 2 + 5, mu_1))
    plt.title("cdf distribution of h(x) for x ~ X|(Y = 1)")
    plt.show()


def q5_iv():
    plt.figure()
    x = np.linspace(0, 1, 1000)
    plt.plot(x, 1 - norm.cdf(logit(x) / 2 + 5, mu_0),
             label="One minus the CDF of h(Z_1) as a function of h(Z_1)")
    plt.plot(x, 1 - norm.cdf(logit(x) / 2 + 5, mu_1),
             label="One minus the CDF of h(Z_2) as a function of h(Z_2)")
    plt.legend(loc='best')
    plt.show()


def q5_v():
    ans = {}
    for t in [0.2, 0.4, 0.55, 0.95]:
        fpr = 1 - norm.cdf(logit(t) / 2 + 5, mu_0)
        tpr = 1 - norm.cdf(logit(t) / 2 + 5, mu_1)
        ans[t] = (fpr, tpr)
    print(ans)


def q5_vi():
    points = []
    for t in [0.2, 0.4, 0.55, 0.95]:
        half = (5 + logit(t) * 0.5)
        points.append(half)
    x = np.linspace(-5, 15, 5000)
    plt.plot(x, norm.pdf(x, mu_0), label='pdf of 0')
    plt.plot(x, norm.pdf(x, mu_1), label='pdf of 1')
    for index, point in enumerate(points):
        plt.plot(point, norm.pdf(point, mu_0), label=point, marker='o')
        plt.plot(point, norm.pdf(point, mu_1), marker='o')

    plt.legend()
    plt.show()


def q5_vii():
    x = np.linspace(0, 1, 1000)
    fpr = []
    tpr = []
    for point in x:
        fpr.append(1 - norm.cdf(logit(point) / 2 + 5, mu_0))
        tpr.append(1 - norm.cdf(logit(point) / 2 + 5, mu_1))
    plt.plot(fpr, tpr)
    plt.show()


# ----------------------------------------------- Q7 -----------------------------------------------

# *** b ***

def q_7_b():

    fpr = [[] for i in range(10)]
    tpr = [[] for i in range(10)]
    loss = []

    for i in range(PROCEDURE_REPETITIONS):
        Logistic_Regression_model_ROC(fpr, i, tpr, x_slice, y_slice, loss)

    print(sum(loss))

    colors = cycle(['Yellow', 'aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'navy',
                    'PeachPuff', 'Sienna', 'Tan', 'Red'])
    for i, color in zip(range(10), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of run {0}'
                                                          ''.format(i))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Q7 - Hands-on (b)')
    plt.legend(loc="lower right")
    plt.show()


def Logistic_Regression_model_ROC(fpr, i, tpr, x, y, loss):
    # Draw 1000 data points from the dataset and keep them aside as a test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1000)
    # Fit a Logistic Regression model on the rest of the data
    lrm = LogisticRegression().fit(x_train, y_train)
    # Use predict proba on the test set and
    pp = lrm.predict_proba(x_test)
    # sort according to the probability of the classifier to predict y = 1
    sorted_pp = y_test[pp[:, 1].argsort()][::-1]  # sort array with regards to 1th column
    NP = np.sum(sorted_pp)  # the number of positives in the test set
    NN = sorted_pp.shape[0] - NP  # the number of negatives in the test set
    N_i = np.cumsum(sorted_pp)
    tpr[i] = [(N_i[j - 1] / NP) for j in range(1, len(N_i))]
    fpr[i] = [((j - N_i[j - 1]) / NN) for j in range(1, len(N_i))]
    loss.append((log_loss(y_test, pp)) / PROCEDURE_REPETITIONS)

# *** c ***

def k_nearest_tarin(k):
    # Draw 1000 data points from the dataset and keep them aside as a test set
    x_train, x_test, y_train, y_test = train_test_split(x_slice, y_slice, test_size=1000)
    x_test_predict = []
    cur_knn = knn(k)
    cur_knn.fit(x_train, y_train)
    # y_pi_(1..k)
    for i in range(len(y_test)):
        x_test_predict.append(cur_knn.predict(x_test[i]))
    return log_loss(y_test, x_test_predict)


def q_7_c():
    loss_dict = {}
    for k in [1, 2, 5, 10, 100]:
        loss = 0
        for i in range(PROCEDURE_REPETITIONS):
            loss += k_nearest_tarin(k)
        loss_dict[k] = (loss / PROCEDURE_REPETITIONS)
        print(loss_dict[k])

# *** d ***
#
# def linear_discriminant_analysis(x_train, x_test, y_train, y_test):
#     train_loss = 0
#     test_loss = 0
#     cur_LDA = LDA()
#     cur_LDA.fit(x_train, y_train)
#
#     for i in range(len(y_train)):
#         if (cur_LDA.predict(x_train[i]) != y_train[i]):
#             train_loss += 1
#     cur_avg_train_loss = train_loss / len(y_train)
#     for i in range(len(y_test)):
#         if (cur_LDA.predict(x_test[i]) != y_test[i]):
#             test_loss += 1
#     cur_avg_test_loss = test_loss / len(y_test)
#     return (cur_avg_train_loss, cur_avg_test_loss)
#
# def quadratic_discriminant_analysis_train(x_train, x_test, y_train, y_test):
#     train_loss = 0
#     test_loss = 0
#     cur_QDA = QDA()
#     cur_QDA.fit(x_train, y_train)
#
#     for i in range(len(y_train)):
#         if (cur_QDA.predict(x_train[i]) != y_train[i]):
#             train_loss += 1
#     cur_avg_train_loss = train_loss / len(y_train)
#     for i in range(len(y_test)):
#         if (cur_QDA.predict(x_test[i]) != y_test[i]):
#             test_loss += 1
#     cur_avg_test_loss = test_loss / len(y_test)
#     return (cur_avg_train_loss, cur_avg_test_loss)
#
#
# def q_7_d():
#     QDA_avg_train_loss = []
#     QDA_avg_test_loss = []
#     for i in range(10):
#         # Draw 1000 data points from the dataset and keep them aside as a test set
#         x_train, x_test, y_train, y_test = train_test_split(x_slice, y_slice, test_size=1000)
#         x_train = x_train[:, DIM]
#         x_test = x_test[:, DIM]
#         QDA_loss_avg = quadratic_discriminant_analysis_train(x_train, x_test, y_train, y_test)
#         QDA_avg_train_loss.append(QDA_loss_avg[0])
#         QDA_avg_test_loss.append(QDA_loss_avg[1])
#     QDA_total_avg_train_loss = sum(QDA_avg_train_loss) / 10
#     QDA_total_avg_test_loss = sum(QDA_avg_test_loss) / 10
#
#     LDA_avg_train_loss = []
#     LDA_avg_test_loss = []
#     for i in range(PROCEDURE_REPETITIONS):
#         # Draw 1000 data points from the dataset and keep them aside as a test set
#         x_train, x_test, y_train, y_test = train_test_split(x_slice, y_slice, test_size=1000)
#         x_train = x_train[:, DIM]
#         x_test = x_test[:, DIM]
#         LDA_loss_avg = linear_discriminant_analysis(x_train, x_test, y_train, y_test)
#         LDA_avg_train_loss.append(LDA_loss_avg[0])
#         LDA_avg_test_loss.append(LDA_loss_avg[1])
#     LDA_total_avg_train_loss = sum(LDA_avg_train_loss) / PROCEDURE_REPETITIONS
#     LDA_total_avg_test_loss = sum(LDA_avg_test_loss) / PROCEDURE_REPETITIONS
#
#     print("LDA - avg_train_loss is :", LDA_total_avg_train_loss)
#     print("QDA - avg_train_loss is :", QDA_total_avg_train_loss)
#     print("LDA - avg_test_loss is :", LDA_total_avg_test_loss)
#     print("QDA - avg_test_loss is :", QDA_total_avg_test_loss)


# ---------------------------------------------- main ----------------------------------------------

def main():
    # Q5
    # q5_i()
    # q5_ii()
    # q5_iii()
    # q5_iv()
    # q5_v()
    # q5_vi()
    # q5_vii()

    # Q7
    q_7_b()
    # q_7_c()
    # q_7_d()
    return


if __name__ == "__main__":
    main()
