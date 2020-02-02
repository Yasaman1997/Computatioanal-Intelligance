import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# A
data = pd.read_csv('data.csv')

# this will create a variable x which has the feature values
x = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values
label = data.iloc[:, 3].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# this will plot the scattered graph of the training set
# plt.scatter(x_test,y_test,c='red')
plt.show()
'''
# print(data)
class1 = data[0:100]
class1_x = class1[0]
class1_y = class1[1]
class1_label = class1[2]

class2 = data[100:]
class2_x = class2[0]
class2_y = class2[1]
class2_label = class2[2]
# print(y)
# print(label)

fig, ax = plt.subplots()
ax.scatter(class1_x, class1_y, label=class1_label,c='purple')
ax.scatter(class2_x, class2_y, label=class2_label, c='green')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''
