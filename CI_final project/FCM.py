import copy
import code
import matplotlib.pyplot as plt
import numpy as np

"""
Parameters:
    _u: membership matrix
    _data:
    _dimension: #features of data
    _fuz: is the hyper- parameter(m)
"""

dimension = 2  # number of features
u = []  # membership matrix 2
old_u = []  # membership matrix 1
cluster_number = 0
# cluster_centers = []
fuzzy_parameter = 2  # m
error = 0.01


class FCM:

    def __init__(self, data, cluster_number, fuzzy_parameter):
        self.cluster_number = cluster_number
        self.m = fuzzy_parameter
        # self.cluster_centers = cluster_centers
        self.data = data
        # U size is len(data) * n_cluster
        self.U_matrix = [[0 for i in range(cluster_number)] for j in range(len(data))]
        # C size is n_cluster * 1
        self.centroid_matrix = None

    def initialized_algorithm(self):
        # select random index for random centroid
        center_indexes = np.random.randint(0, len(self.data) - 1, self.cluster_number)
        self.centroid_matrix = [self.data[i][0] for i in center_indexes]

    def distance(self, A, B):
        L = 0
        for i in range(len(A)):
            L += (A[i] - B[i]) ** 2
        return L ** 0.5


    # def u(self):
    #     return self._u
    #
    # def data(self, d=None):
    #     """getter and setter of data"""
    #     if d:
    #         self._data = d
    #     return self._data

    # def cluster_centers(self):
    #     return self._cluster_centers

    # def check_membership(data):
    #     for i in range(len(data)):
    #         R = 0
    #         old_u.append([])
    #         u.append([])
    #         for j in range(cluster_number):
    #             u[i].append(random.uniform(min(data[:, 0:1]), max(data[:, 0:1])))
    #             R += u[i][j]
    #
    #         for j in range(cluster_number):
    #             old_u[i].append(0)
    #             u[i][j] /= R

    # uik:belong Xi to Ck
    def update_membership(self):
        for i in range(len(self.data)):
            for j in range(len(self.centroid_matrix)):
                T1 = 0
                T2 = float((self.distance(self.data[i][0], self.centroid_matrix[j])))
                for center in self.centroid_matrix:
                    T3 = float(self.distance(self.data[i][0], center))
                    T1 += float(T2 / T3) ** (2 / (self.m - 1))
                self.u_matrix[i][j] = 1 / T1

    # compute and update centroid of clusters
    def update_centroid(self):
        for j in range(len(self.centroid_matrix)):
            u = 0
            cx, cy = 0
            for i in range(len(self.data)):
                cx += copy.deepcopy((self.u_matrix[i][j] ** self.m) * self.data[i][0][0])
                cy += copy.deepcopy((self.u_matrix[i][j] ** self.m) * self.data[i][0][1])
                u += self.u_matrix[i][j] ** self.m
            self.centroid_matrix[j][0] = copy.deepcopy(cx / u)
            self.centroid_matrix[j][1] = copy.deepcopy(cy / u)

    def clustering(self):
        i = 200
        first = True
        while i > 0:
            if first:
                self.initialized_algorithm()
            else:
                self.update_centroid()
            self.update_membership()
            first = False
            i -= 1
        return [self.u_matrix, self.centroid_matrix]

    # def update_membership(data):
    #     U = 0
    #     for i in range(len(data)):
    #         for j in range(cluster_number):
    #             R = 0
    #             old_u[i][j] = u[i][j]
    #             d1 = distance(data[i], cluster_centers[j])
    #             for k in range(cluster_number):
    #                 d2 = distance(data[i], cluster_centers[k])
    #                 R += (d1 / d2) ** (2 / (fuzzy_parameter - 1))
    #             u[i][j] = 1 / R
    #             U.append(u[i][j])
    #         return U

    # def calculate_cluster_centers():
    #     for i in range(cluster_number):
    #         final_center = []
    #         for j in range(dimension):
    #             R1, Temp = 0
    #             c = []
    #             for k in range(len(data)):
    #                 Temp += u[k][i] ** fuzzy_parameter
    #                 R1 += Temp * data[k][j]
    #                 final_center.append(R1 / Temp)
    #         cluster_centers.append(final_center)
    #         return cluster_centers

    # def clustering(cluster_number, iteration, data):
    #     check_membership(data)
    #
    #     for i in range(iteration):
    #         update_centroid()
    #         update_membership(data)
    #
    #         if convergence() < error:
    #             break

    # def convergence():
    #     return abs(la.norm(u, ord=2) - la.norm(old_u, ord=2))


def run():
    path = 'D:/Computational intelligence/CI_final project/4clstrain1200.csv'
    fcm = FCM(4, code.prepare_data(path), 0.1)
    fcm.clustering()
    fig, ax = plt.subplots(1, 1, True, True)
    plt.show()

# cluster_num = input("Enter number of cluster: ")
#
# prepare_data()
# clustering(cluster_num, 100, data)
#
# plt.plot(data.iloc[:, 0:1].values, data.iloc[:, 1], 'bo')
# for cl in cluster_centers():
#     plt.plot(cl[0], cl[1], 'r*')
# plt.grid(b=None, which='both', axis='both', color='gray', linestyle='-', linewidth=2)
# plt.title('number of clusters: ' + cluster_num + ', number of data: ' + len(data))
# plt.show()
run()