import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# A
# def read():
#     data = pd.read_csv('Dataset1.csv', header=None)
#
#     # this will create a variable x which has the feature values
#     x = data.iloc[:, 0:1].values
#     y = data.iloc[:, 1].values
#     label = data.iloc[:, 2].values
#
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=None, shuffle=True)
#
#     # this will plot the scattered graph of the training set
#     print(data)
#     fig, ax = plt.subplots()
#     ax.scatter(x_test, y_test, c=label)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.show()


def prepare_data(path):
    x0 = []
    y0 = []
    x1 = []
    y1 = []

    path='Dataset1.csv'
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
    #plt.scatter(x0, y0, color="red")
    #plt.scatter(x1, y1, color="green")
    #plt.title('data')
   # plt.show()
    return dataset_values




prepare_data('2clstrain1200.csv')
