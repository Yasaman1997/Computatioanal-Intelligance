import csv
import math

import numpy as np
from matplotlib import pyplot as plt

n_epoch = 3000


def sigmoid(x):
    return (1 / (1 + math.exp(-x)))


def compute_Z(W, V, X, b0, b1):
    z = [sigmoid(np.dot(W, X) + b0), sigmoid(np.dot(V, X) + b1)]
    return z


def compute_y(W, U, V, X, b0, b1, b2):
    Z = compute_Z(W, V, X, b0, b1)
    y = sigmoid(np.dot(U, Z) + b2)
    return y


def compute_W(W, U, V, X, b0, b1, b2, y0, weight):
    Z = compute_Z(W, V, X, b0, b1)
    y = compute_y(W, U, V, X, b0, b1, b2)

    sigmoid_gradient1 = Z[0] * (1 - Z[0])
    sigmoid_gradient2 = y * (1 - y)

    if weight == 1:
        return U[0] * X[1] * (y - y0) * sigmoid_gradient1 * sigmoid_gradient2
    elif weight == 0:
        return U[0] * X[0] * (y - y0) * sigmoid_gradient1 * sigmoid_gradient2

    else:
        return U[0] * (y - y0) * sigmoid_gradient1 * sigmoid_gradient2


def compute_V(W, U, V, X, b0, b1, b2, y0, weight):
    Z = compute_Z(W, V, X, b0, b1)
    y = compute_y(W, U, V, X, b0, b1, b2)

    sigmoid_gradient1 = Z[1] * (1 - Z[1])
    sigmoid_gradient2 = y * (1 - y)

    if weight == 1:
        return U[0] * X[1] * (y - y0) * sigmoid_gradient1 * sigmoid_gradient2
    elif weight == 0:
        return U[0] * X[0] * (y - y0) * sigmoid_gradient1 * sigmoid_gradient2

    else:
        return U[0] * (y - y0) * sigmoid_gradient1 * sigmoid_gradient2


def compute_U(W, V, U, X, b0, b1, b2, y0, weight):
    Z = compute_Z(W, V, X, b0, b1)
    y = compute_y(W, U, V, X, b0, b1, b2)

    sigmoid_new = y * (1 - y)

    if weight == 1:
        return Z[1] * (y - y0) * sigmoid_new
    elif weight == 0:
        return Z[0] * (y - y0) * sigmoid_new
    else:
        return (y - y0) * sigmoid_new


def train(label):
    error = 0
    b = [np.random.normal(0, 1) for i in range(0, 3)]
    Ir = 3 / len(label)

    W = [np.random.normal(0, 1) for i in range(0, 2)]
    U = [np.random.normal(0, 1) for i in range(0, 2)]
    V = [np.random.normal(0, 1) for i in range(0, 2)]

    gradient_w = [0, 0, 0]
    gradient_v = [0, 0, 0]
    gradient_u = [0, 0, 0]

    for i in range(0, n_epoch):

        for X in label:
            y = compute_y(W, U, V, X[0], b[0], b[1], b[2])
            y0 = sigmoid(X[1])
            error += 0.5 * ((y - y0) ** 2)

            for w in range(0, 3):
                for X in label:
                    gradient_w[w] = 0
                    gradient_w[w] += compute_W(W, U, V, X[0], b[0], b[1], b[2], X[1], w)

            for v in range(0, 3):
                for X in label:
                    gradient_v[v] = 0
                    gradient_v[v] += compute_V(W, U, V, X[0], b[0], b[1], b[2], X[1], v)

            for u in range(0, 3):
                for X in label:
                    gradient_u[u] = 0
                    gradient_u[u] += compute_U(W, U, V, X[0], b[0], b[1], b[2], X[1], u)

            for j in range(0, 2):
                W[j] -= Ir * gradient_w[j]
                V[j] -= Ir * gradient_v[j]
                U[j] -= Ir * gradient_u[j]

            b[0] -= Ir * gradient_w[2]
            b[1] -= Ir * gradient_v[2]
            b[2] -= Ir * gradient_u[2]

    return [W, V, U, b]


def get_labels():
    x0 = []
    x1 = []
    y0 = []
    y1 = []

    dataset = csv.reader(open("data.csv"))
    dataset_label = []

    for col in dataset:
        dataset_label.append(
            [[float(col[0].replace("'", "")), float(col[1].replace("'", ""))], int(col[2].replace("'", ""))])
        if int(col[2].replace("'", "")) == 0:
            x0.append(float(col[0].replace("'", "")))
            y0.append(float(col[1].replace("'", "")))
        else:
            x1.append(float(col[0].replace("'", "")))
            y1.append(float(col[1].replace("'", "")))

    return dataset_label


def model(W, V, U, b, label_test):
    for X in label_test:
        Z = compute_Z(W, V, X[0], b[0], b[1])
        Y = compute_y(W, V, U, X[0], b[0], b[1], b[2])
        if Y > 0.5:
            X[1] = 1
        else:
            X[1] = 0
    return label_test


def boundary_plot(W=None, V=None, U=None, b=None):
    plt.xlim()
    plt.ylim()
    x_range = np.arange(-1, 1, 0.1)
    y_range = np.arange(-1, 1, 0.1)

    xx, yy = np.meshgrid(x_range, y_range)
    cmap = plt.get_cmap('Paired')
    zz = np.zeros(xx.shape)

    for i in range(zz.shape[0]):
        for j in range(zz.shape[1]):
            x_vector = [[[xx[i][j], yy[i][j]], 0]]

            network_answer = model(W, V, U, b, x_vector)
            zz[i][j] = network_answer[0][1]

    plt.pcolormesh(xx, yy, zz, cmap=cmap)


y0 = []
y1 = []
x0 = []
x1 = []
label_test = []
label_train = []
dataset = get_labels()
np.random.shuffle(dataset)

for i in range(0, len(dataset)):
    if i < np.round(0.8 * len(dataset)):
        label_train.append([[dataset[i][0][0], dataset[i][0][1]], dataset[i][1]])
    else:
        label_test.append([[dataset[i][0][0], dataset[i][0][1]], dataset[i][1]])
W, V, U, b = train(label_train)
#boundary_plot(W, V, U, b)
label_test = model(W, V, U, b, label_train)

for i in range(0, len(label_test)):
    if label_test[i][1] == 0:
        x0.append(label_test[i][0][0])
        y0.append(label_test[i][0][1])
    else:
        x1.append(label_test[i][0][0])
        y1.append(label_test[i][0][1])

plt.scatter(x0, y0, color="blue")
plt.scatter(x1, y1, color="red")
plt.show()
