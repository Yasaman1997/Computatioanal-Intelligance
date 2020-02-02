# from mlxtend.general_plotting import category_scatter
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# A
data = pd.read_csv('data.csv', header=None)

# print(data)
x = data[0]
y = data[1]
label = data[2]
# print(x)
# print(y)
# print(label)

fig, ax = plt.subplots()
ax.scatter(x, y, c=label)
plt.xlabel('x')
plt.ylabel('y')

plt.show()
plt.savefig('ScatterPlot_1_A.png')

# fig =category_scatter(x='x', y='y', label_col='label', data=data, legend_loc='upper left')

# B
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, shuffle=True)


fig, ax = plt.subplots()
#ax.scatter(X_train, y_train, c=label)
#ax.scatter(X_test, y_test,c=label)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
