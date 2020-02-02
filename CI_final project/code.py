import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# A
def read():
    data = pd.read_csv('2clstrain1200.csv', header=None)

    # this will create a variable x which has the feature values
    x = data.iloc[:, 0:1].values
    y = data.iloc[:, 1].values
    label = data.iloc[:, 2].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    # this will plot the scattered graph of the training set
    # plt.scatter(x_test,y_test,c='red')
    # plt.show()

    # print(data)
    class1 = data[0:300]
    class1_x = class1[0]
    class1_y = class1[1]
    class1_label = class1[2]

    class2 = data[300:]
    class2_x = class2[0]
    class2_y = class2[1]
    class2_label = class2[2]
    # print(y)
    # print(label)

    fig, ax = plt.subplots()
    # ax.scatter(class1_x, class1_y, label=class1_label, c='purple')
    # ax.scatter(class2_x, class2_y, label=class2_label, c='green')
    #ax.scatter(class1_x, class1_y, c=class1_label)
    #ax.scatter(class2_x, class2_y, c=class2_label)
    ax.scatter(x, y, c=label)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()



def prepare_data():
    x0 = []
    y0 = []
    x1 = []
    y1 = []

    path = '2clstrain1200.csv'
    dataset = open(path, 'r').read()
    # dataset = csv.reader(dataset_file)
    dataset_values = []
    label = []

    dataset = dataset.split('\n')
    for row in dataset:
        data = row.split(',')

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
    plt.scatter(x1, y1, color="red")
    #plt.show()
    return dataset_values


#prepare_data()
#read()
