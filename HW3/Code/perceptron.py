import csv
import math

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return (1 / (1 + math.exp(-x)))


def compute_gradient(W, X, b, y0, weight):
    y = compute_y(W, X, b)
    sigmoid_new = y * (1 - y)
    if weight == 0:
        return X[0] * (y - y0) * sigmoid_new
    elif weight == 1:
        return X[1] * (y - y0) * sigmoid_new
    else:
        return (y - y0) * sigmoid_new


def compute_y(W, X, b):
    return sigmoid(np.dot(X, W) + b)


def get_labels():
    dataset = csv.reader(open("data.csv"))
    dataset_label = []

    x0 = []
    x1 = []
    y0 = []
    y1 = []

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


def train(train_label):
    n_epoch = 3000
    Ir = 3 / len(train_label)  # learning_rate
    gradient = [0, 0, 0]
    W = []
    b = np.random.normal(0, 1)
    W.append(np.random.normal(0, 1))
    W.append(np.random.normal(0, 1))

    for i in range(0, n_epoch):
        for w in range(0, 3):
            gradient[w] = 0
            for X in train_label:
                gradient[w] += compute_gradient(W, X[0], b, X[1], w)
        for w in range(0, 2):
            W[w] -= Ir * gradient[w]
            b -= Ir * gradient[2]
    return [W, b]


def model(W, b, test_label):
    for X in test_label:
        # Y = sigmoid(np.dot(W, X[0]) + b)
        Y = compute_y(W, X[0], b)
        if Y >= 0.5:
            X[1] = 1
        else:
            X[1] = 0
    return test_label


x0 = []
x1 = []
y0 = []
y1 = []

x0_predicted = []
x1_predicted = []
y0_predicted = []
y1_predicted = []

dataset = get_labels()
np.random.shuffle(dataset)

train_value = []
test_value = []

# plot test data
for i in range(0, len(dataset)):
    # split
    if i < np.round(0.8 * len(dataset)):
        train_value.append([[dataset[i][0][0], dataset[i][0][1]], dataset[i][1]])
    else:
        test_value.append([[dataset[i][0][0], dataset[i][0][1]], dataset[i][1]])

for i in range(0, len(test_value)):
    if test_value[i][1] == 0:
        x0.append(test_value[i][0][0])
        y0.append(test_value[i][0][1])
    else:
        x1.append(test_value[i][0][0])
        y1.append(test_value[i][0][1])

plt.scatter(x0, y0, color="blue")
plt.scatter(x1, y1, color="red")
plt.show()
# print(len(test_value))
# print(len(dataset))

W, b = train(train_value)
predicted_label_test = model(W, b, test_value)

for i in range(0, len(predicted_label_test)):
    if predicted_label_test[i][1] == 0:
        x0_predicted.append(predicted_label_test[i][0][0])
        y0_predicted.append(predicted_label_test[i][0][1])
    else:
        x1_predicted.append(predicted_label_test[i][0][0])
        y1_predicted.append(predicted_label_test[i][0][1])

plt.scatter(x0_predicted, y0_predicted, color="blue")
plt.scatter(x1_predicted, y1_predicted, color="red")
plt.show()

correct_pred = 0

for i in range(0, len(predicted_label_test)):
    if predicted_label_test[i][1] == test_value[i][1]:
        correct_pred += 1

accuracy = correct_pred / len(test_value)
print(accuracy)
