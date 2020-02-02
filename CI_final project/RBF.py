import math

import matplotlib.pyplot as plt
import numpy as np

import FCM

centroid = []


class RBF:
    def __init__(self, path, num_of_clusters, fuzzy_parameter, landa, n_class):
        path = path
        self.m = fuzzy_parameter
        self.n_cluster = num_of_clusters
        self.n_class = n_class
        self.dataset = RBF.read_file(path)
        self.raduis = landa
        self.G_matrix = np.array([[0.0 for i in range(self.n_cluster)] for j in range(int(len(self.dataset) * 0.7))])
        self.Y_matrix = np.array([[0 for i in range(self.n_class)] for j in range(int(len(self.dataset) * 0.7))])
        self.W_matrix = None
        self.G_matrix_test = np.array(
            [[0.0 for i in range(self.n_cluster)] for j in range(int(len(self.dataset) - len(self.dataset) * 0.7))])
        self.Y = [0.0 for i in range(int(len(self.dataset) - len(self.dataset) * 0.7))]
        self.output = np.array([[0.0 for i in range(self.n_class)] for j in range(int(len(self.dataset) * 0.7))])

    def distance(self, A, B):
        L = 0
        for i in range(len(A)):
            L += (A[i] - B[i]) ** 2
        return L ** 0.5

    def read_file(path):
        x0 = []
        y0 = []
        x1 = []
        y1 = []

        dataset = open(path, 'r').read()
        dataset_values = []
        label = []

        dataset = dataset.split('\n')
        for row in dataset:
            data = row.split(',')
            # print(data)
            if (len(data) > 1):
                dataset_values.append([[float(data[0].replace("'", "")),
                                        float(data[1].replace("'", ""))
                                        ], int(data[2].replace("'", "").replace('0', '').replace('.', ''))])
                label.append(int(data[2].replace("'", "").replace('0', '').replace('.', '')))
                if int(data[2].replace("'", "").replace('0', '').replace('.', '')) == 1:
                    y0.append(float(data[1].replace("'", "")))
                    x0.append(float(data[0].replace("'", "")))
                else:
                    y1.append(float(data[1].replace("'", "")))
                    x1.append(float(data[0].replace("'", "")))

        open(path, 'r').close()
        plt.scatter(x0, y0, color="blue")
        plt.scatter(x1, y1, color="orange")
        plt.show()
        return dataset_values

    def u(self):
        return self._u

    # uik:memberhip of Xi to Ck
    def update_membership(self, x, vi):
        T1 = float((self.distance(x, vi)))
        T2, T3 = 0
        for center in self.centroid_matrix:
            T2 = float(self.distance(x, center))
            T3 = (float(T1 / T2) ** (2 / (self.m - 1)))

        uik = 1 / T3
        return uik

    def compute_G(self, start, end, G):
        for i in range(len(self.centroid_matrix)):
            uik = 0
            ci = np.array([[0.0, 0.0], [0.0, 0.0]])

            for j in range(start, end):
                if G == 0:
                    u = self.U_matrix[j - start][i]
                else:
                    u = self.update_membership(self.dataset[j][0], self.centroid_matrix[i])

                Temp = np.array([u ** self.m * self.dataset[j][0][0], u ** self.m * self.dataset[j][0][1]]) - \
                       np.array([u ** self.m * float(self.centroid_matrix[i][0]),
                                 u ** self.m * float(self.centroid_matrix[i][1])])

                ci += [[Temp[0] ** 2, Temp[0] * Temp[1]], [Temp[0] * Temp[1], Temp[1] ** 2]]
                uik += (u ** self.m)

            ci = ci / uik
            centroid.append(ci)

            for j in range(start, end):
                x = np.array([self.dataset[j][0][0], self.dataset[j][0][1]])

                if G == 0:
                    self.G_matrix[j - start][i] = math.exp(-self.raduis * np.matmul(
                        np.matmul(np.transpose(x - self.centroid_matrix[i]), np.linalg.inv(ci)),
                        x - self.centroid_matrix[i]))
                else:
                    self.G_matrix_test[j - start][i] = math.exp(
                        -self.c_raduis * np.matmul(np.matmul(np.transpose(x - self.centroid_matrix[i]),
                                                             np.linalg.inv(ci)),
                                                   x - self.centroid_matrix[i]))

    def rbf_train(self):
        np.random.shuffle(self.dataset)
        for i in range(int(len(self.dataset) * 0.7)):
            self.Y_matrix[i][self.dataset[i][1] - 1] = 1
        fcm = FCM.FCM(self.n_cluster, self.dataset[0:int(len(self.dataset) * 0.7)], self.m)
        self.U_matrix, self.centroid_matrix_ = fcm.clustering()
        self.compute_G(0, int(len(self.dataset) * 0.7), 0)
        self.W_matrix = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(self.G_matrix),
                                                                    self.G_matrix)), np.transpose(self.G_matrix)),
                                  self.Y_matrix)

    def rbf_test(self):
        self.compute_G(int(len(self.dataset) * 0.7) + 1, len(self.dataset), 1)
        self.output = np.matmul(self.G_matrix_test, self.W_matrix)
        for i in range(len(self.output)):
            self.Y[i] = np.argmax(self.output[i]) + 1

    def Evaluation(self):
        start = int(len(self.dataset) * 0.7) + 1
        end = len(self.dataset)

        sum = 0.0

        for i in range(start, end):
            e = self.dataset[i][1] - self.Y[i - start]
            if e > 0 or e < 0:
                sum += 1
        length = abs(len(self.dataset) * 0.3)
        accuracy = 1 - sum / length
        return accuracy


def run():
    for i in range(2, 32, 4):
        rbf = RBF("2clstrain1200.csv", 3, 2, 1, 2)
        rbf.rbf_train()
        rbf.rbf_test()
        plt.scatter(
            [rbf.centroid_matrix[0][0], rbf.centroid_matrix[1][0], rbf.centroid_matrix[2][0], rbf.centroid_matrix[3][0],
             rbf.centroid_matrix[4][0],
             rbf.centroid_matrix[5][0], rbf.centroid_matrix[6][0], rbf.centroid_matrix[7][0]],
            [rbf.centroid_matrix[0][1], rbf.centroid_matrix[1][1], rbf.centroid_matrix[2][1], rbf.centroid_matrix[3][1],
             rbf.centroid_matrix[4][1],
             rbf.centroid_matrix[5][1], rbf.centroid_matrix[6][1], rbf.centroid_matrix[7][1]], color='blue')
        plt.show()
        print(rbf.Evaluation())
        #print(centroid_matrix)
