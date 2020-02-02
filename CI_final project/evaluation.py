import numpy as np


def accuracy(y, y_hat, data):
    n = 0
    for i in range(len(data)):
        n += np.abs(np.sign(y - y_hat))
        n = n / len(data)
        n = 1 - n

def G(data,vi,landa,c):
    np.exp(-landa * (np.transpose(data - vi)))