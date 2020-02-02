import matplotlib.pyplot as plt
import numpy as  np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import hw2.Code.file_handler as fh

#data=fh.prepare_data('Dataset1.csv')
#data = fh.read_from_file('Dataset2.csv')

# split into training and testing sets


# X_train= X_train.reshape(-1,1)

# standardize the features
sc = StandardScaler()
#data_std = sc.fit_transform(data)

# intialize pca and logistic regression model
pca = PCA(n_components=2)

# fit and transform data
#data_pca = pca.fit_transform(data_std)

data=pd.read_csv('Dataset1.csv',header=None)

X1 = data.iloc[:, 0]
Y1 = data.iloc[:, 1]
X2 = data.iloc[:, 0]
Y2 = data.iloc[:, 1]

plt.scatter(X1, Y1)
plt.scatter(X2, Y2)
plt.show()
plt.savefig('data2.jpg')
#np.savetxt("data2_after_pca_lib.csv", data_pca, delimiter=",")
