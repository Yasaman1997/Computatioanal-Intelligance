import copy
import math

import matplotlib.pyplot as plt
import numpy as np

import code

i = 0
j = 0


class FCM:
    def __init__(self, n_cluster, data, fuzziness_parameter):
        self.n_cluster = n_cluster
        self.data = data
        self.m = fuzziness_parameter
        self.centroid_matrix = None
        self.U_matrix = [[0 for i in range(n_cluster)] for j in range(len(data))]

    def random_centroid(self):
        center_indexes = np.random.randint(0, len(self.data) - 1, self.n_cluster)
        self.centroid_matrix = [self.data[i][0] for i in center_indexes]

    def distance(self, A, B):
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(A, B)))
        if d == 0:
            return 0.00000000000001
        else:
            return d

    # uik: how much  Xi belongs to Ck
    def compute_u(self):
        u = []
        for i in range(len(self.data)):
            for j in range(len(self.centroid_matrix)):
                T1 = 0
                T2 = float((self.distance(self.data[i][0], self.centroid_matrix[j])))
                for ck in self.centroid_matrix:
                    T3 = float(self.distance(self.data[i][0], ck))
                    T1 += pow(float(T2 / T3), 2 / (self.m - 1))
                self.U_matrix[i][j] = 1 / T1
                u.append(self.U_matrix[i][j])
        # print('******************U:****************')
        # print(self.U_matrix)
        # np.savetxt("U.txt", u)

    # compute and update centroid of clusters .
    def compute_c(self):
        for i in range(len(self.centroid_matrix)):
            u = 0
            cx = 0
            cy = 0
            for j in range(len(self.data)):
                u += self.U_matrix[j][i] ** self.m
                cx += copy.deepcopy((self.U_matrix[j][i] ** self.m) * self.data[j][0][0])
                cy += copy.deepcopy((self.U_matrix[j][i] ** self.m) * self.data[j][0][1])
            self.centroid_matrix[i][0] = copy.deepcopy(cx / u)
            self.centroid_matrix[i][1] = copy.deepcopy(cy / u)
        # print('******************C:****************')
        # print(self.centroid_matrix)
        # np.savetxt("C.txt", self.centroid_matrix)

    def clustering_algorithm(self):
        i = 200
        begin = True
        while i > 0:

            if begin:
                self.random_centroid()
            else:
                self.compute_c()
            self.compute_u()
            begin = False
            i -= 1
        print('U:')
        print(self.U_matrix)
        print('C:')
        print(self.centroid_matrix)
        np.savetxt("C.txt", self.centroid_matrix)
        np.savetxt("U.txt", self.U_matrix)

        return [self.U_matrix, self.centroid_matrix]


def run():
    fcm = FCM(10, code.prepare_data(), 10)
    fcm.clustering_algorithm()

    # print(type(fcm.centroid_matrix))

    plt.scatter(*zip(*fcm.centroid_matrix), c='black')
    plt.show()


run()
