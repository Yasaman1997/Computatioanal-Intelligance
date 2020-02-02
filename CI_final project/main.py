import matplotlib.pyplot as plt
import FCM as fcm
from sklearn.datasets.samples_generator import make_blobs

cluster_num = input("Enter number of cluster: ")
data_num = input("Enter number of data(data set size): ")


fcm.create_random_data(int(data_num), 2, int(cluster_num))
fcm.exec(int(cluster_num), 100, fcm.data())

plt.plot(fcm.data()[:, 0], fcm.data()[:, 1], 'bo')
for ct in fcm.cluster_centers():
    plt.plot(ct[0], ct[1], 'r*')
plt.grid(b=None, which='both', axis='both', color='gray', linestyle='-', linewidth=2)
plt.title('number of clusters: ' + cluster_num + ', number of data: ' + data_num)
plt.show()
