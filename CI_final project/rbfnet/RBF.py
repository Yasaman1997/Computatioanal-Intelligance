import math

import matplotlib.pyplot as plt
import numpy as np

import FCM
import code


class RBF:
    def __init__(self, path, clusters, fuzziness_parameter, gama, n_class):
        path = path
        self.dataset = code.prepare_data()
        self.n_cluster = clusters
        self.m = fuzziness_parameter
        self.n_class = n_class
        self.c_raduis = gama
        self.G_matrix = np.array([[0.0 for i in range(self.n_cluster)] for j in range(int(len(self.dataset) * 0.7))])
        self.Y_matrix = np.array([[0 for i in range(self.n_class)] for j in range(int(len(self.dataset) * 0.7))])
        self.W_matrix = None
        self.G_matrix_test = np.array(
            [[0.0 for i in range(self.n_cluster)] for j in range(int(len(self.dataset) - len(self.dataset) * 0.7))])
        self.Y = [0.0 for i in range(int(len(self.dataset) - len(self.dataset) * 0.7))]
        self.Output_matrix = np.array([[0.0 for i in range(self.n_class)] for j in range(int(len(self.dataset) * 0.7))])

    def distance(self, point1, point2):
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
        if d == 0:
            return 0.00000001
        else:
            return d

    def get_uik(self, x, vi):
        T1 = 0
        T2 = float((self.distance(x, vi)))
        for ck in self.C_matrix:
            T3 = float(self.distance(x, ck))
            T1 += pow(float(T2 / T3), 2 / (self.m - 1))
        uik = 1 / T1
        return uik

    def compute_G(self, start, end, G):
        g1 = []
        g2 = []
        for i in range(len(self.C_matrix)):
            ci = np.array([[0.0, 0.0],
                           [0.0, 0.0]])
            uik = 0
            u = 0
            for j in range(start, end):
                if G == 0:
                    u = self.U_matrix[j - start][i]
                else:
                    u = self.get_uik(self.dataset[j][0], self.C_matrix[i])
                g = np.array([u ** self.m * self.dataset[j][0][0],
                              u ** self.m * self.dataset[j][0][1]]) - \
                    np.array([u ** self.m * float(self.C_matrix[i][0]),
                              u ** self.m * float(self.C_matrix[i][1])])

                ci += [[g[0] ** 2, g[0] * g[1]], [g[0] * g[1], g[1] ** 2]]
                uik += (u ** self.m)
            ci = ci / uik

            for j in range(start, end):
                x = np.array([self.dataset[j][0][0],
                              self.dataset[j][0][1]])
                if G == 0:
                    self.G_matrix[j - start][i] = math.exp(
                        -self.c_raduis * np.matmul(np.matmul(np.transpose(x - self.C_matrix[i]),
                                                             np.linalg.inv(ci)),
                                                   x - self.C_matrix[i]))
                    # g1.append(self.G_matrix)
                    # np.savetxt("G1.txt", g1)
                else:
                    self.G_matrix_test[j - start][i] = math.exp(
                        -self.c_raduis * np.matmul(np.matmul(np.transpose(x - self.C_matrix[i]),
                                                             np.linalg.inv(ci)),
                                                   x - self.C_matrix[i]))
                    # g2.append(self.G_matrix_test)
                    # np.savetxt("G2.txt", g2)

    def Run_Rbf(self):
        np.random.shuffle(self.dataset)
        for i in range(int(len(self.dataset) * 0.7)):
            self.Y_matrix[i][self.dataset[i][1] - 1] = 1
        fcm = FCM.FCM(self.n_cluster, self.dataset[0:int(len(self.dataset) * 0.7)], self.m)  # ue FCM
        self.U_matrix, self.C_matrix = fcm.clustering_algorithm()
        self.compute_G(0, int(len(self.dataset) * 0.7), 0)
        self.W_matrix = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(self.G_matrix),
                                                                    self.G_matrix)), np.transpose(self.G_matrix)),
                                  self.Y_matrix)
        self.Output_matrix = np.matmul(self.G_matrix, self.W_matrix)
        print('W_matrix:')
        print(self.W_matrix)
        print('output:')
        print(self.Output_matrix)

    def rbf_test(self):
        self.compute_G(int(len(self.dataset) * 0.7) + 1, len(self.dataset), 1)
        self.Output_matrix = np.matmul(self.G_matrix_test, self.W_matrix)
        # print(self.dataset[int(len(self.dataset) * 0.7)+1:len(self.dataset)])
        for i in range(len(self.Output_matrix)):
            self.Y[i] = np.argmax(self.Output_matrix[i]) + 1
        print('y:')
        print(self.Y)
        print('predicted_output:')
        print(self.Output_matrix)

    def accuracy(self):
        sum = 0.0
        acc = []
        start = int(len(self.dataset) * 0.7) + 1
        end = len(self.dataset)
        for i in range(start, end):
            dif = self.dataset[i][1] - self.Y[i - start]
            # plt.scatter(self.Y[i - start], c='green')
            # plt.scatter(self.dataset[i][1], c='red')
            plt.show()
            if dif > 0 or dif < 0:
                sum += 1
        accuracy = 1 - sum / int(len(self.dataset) * 0.3)
        acc.append(accuracy)
        np.savetxt("acc.txt", acc)
        print('accuracy:')
        print(accuracy)


def run():
    for i in range(2, 32, 2):
        rbf = RBF("2clstrain1200.csv", 10, 2, 1, 2)
        rbf.Run_Rbf()
        rbf.rbf_test()
        plt.scatter([rbf.C_matrix[0][0], rbf.C_matrix[1][0], rbf.C_matrix[2][0], rbf.C_matrix[3][0], rbf.C_matrix[4][0],
                     rbf.C_matrix[5][0], rbf.C_matrix[6][0], rbf.C_matrix[7][0]],
                    [rbf.C_matrix[0][1], rbf.C_matrix[1][1], rbf.C_matrix[2][1], rbf.C_matrix[3][1], rbf.C_matrix[4][1],
                     rbf.C_matrix[5][1], rbf.C_matrix[6][1], rbf.C_matrix[7][1]], color='black')
        plt.show()
        # print('accuracy:')
        print(rbf.accuracy())


run()
