# import statistics
#
# import matplotlib.pyplot as plt
#
# import hw2.Code.file_handler as fh
#
#
# def plot():
#     """
#     Plot data points with the best vector for dimension reduction
#     :return:
#     """
#
#
# data = fh.read_from_file('Dataset2.csv')
# X = data.iloc[:, 0]
# Y = data.iloc[:, 1]
# Z = []
# axis = []
#
# i = 0
# for x in X:
#     zi = 5 * x + 7 * Y[i]
#     Z.append(zi)
#     axis.append((zi - 0.01 * x) / -0.01)
#     i += 1
#
# Zprim = statistics.stdev(Z, statistics.mean(Z))
# print(Zprim)
#
# plt.scatter(X, Y)
# plt.plot(Z)
# plt.show()
#


import sys

sys.path.insert(0, sys.path[0] + "..")
# if __name__ == '__main__':

import os.path
import statistics
import numpy as np
from matplotlib import pyplot as plt

libdir = os.path.dirname(__file__)
sys.path.append(os.path.split(libdir)[0])
import hw2.Code.file_handler as fh
from hw2.Code import ES
from numpy import linalg as eign
# import     fileTool.file_handler

# if __name__ == '__main__':
# plt.plot(file_handler.load_data()[0],file_handler.load_data()[1])
# plt.show()

data = fh.load_data()
X = data[0]
Y = data[1]
Z = []
axis = []
# arr=[][]
i = 0
for x in X:
    zi = 0.01 * x - 0.01 * Y[i]
    Z.append(zi)
    axis.append((zi - 0.01 * x) / -0.01)
    i += 1

print(statistics.stdev(Z, statistics.mean(Z)))
print('Z')
print(Z)
#eign.eig([X,Y])
plt.scatter(X, Y, )
plt.scatter(X,Y)
plt.show()
print('np.random.randint:')
print(np.random.randint(0, 9))
X.sort(key=lambda x: x, reverse=True)
print(sorted(X, key=lambda x: x, reverse=False))
es = ES.ES(-0.1, 0.1, data, 40, 20, 1, 0.4, 0.3)
es.evolution_process()
es.initial_population()
# print(es.initial_population_set[5].score)
es.select_parent()
print(es.select_population_set[52].score)
es.mutation()
print(es.select_population_set[52].score)
# print(np.round(2.3))
es.re_combination()
es.select_next_population()
for i in range(0, 10):
    print(es.initial_population_set[i].score)


def goal_st_derivation(X_dim, Y_dim):
    X_mean = statistics.mean(X_dim)
    Y_mean = statistics.mean(Y_dim)
    co_variance_matrix = [][2]
    co_variance_matrix[0][0] = statistics.variance(X_dim, X_mean)
    co_variance_matrix[1][1] = statistics.variance(Y_dim, Y_mean)

    i = 0
    co_variance = 0
    for index in X_dim:
        co_variance += (index - X_mean) * (Y_dim[i] - Y_mean)
    co_variance /= (len(X_dim) - 1)
    co_variance_matrix[0][1] = co_variance
    co_variance_matrix[1][0] = co_variance
    plt.scatter(co_variance_matrix[0][1],co_variance_matrix[1][0])
    plt.show()
