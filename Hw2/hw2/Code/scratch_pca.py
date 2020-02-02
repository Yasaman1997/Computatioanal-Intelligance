import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import file_handler as fh

# data=fh.read_from_file('Dataset1.csv')
data = fh.read_from_file('Dataset1.csv')


# standardize the features
sc = StandardScaler()
data_std = sc.fit_transform(data)


# X_train= X_train.reshape(-1,1)

# standardize the features


cov_mat = np.cov(data_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# calculate cumulative sum of explained variances
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# plot explained variances
plt.bar(range(1, 14), var_exp, alpha=0.5,
        align='center', label='individual explained variance')
plt.step(range(1, 14), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.Ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.show()

