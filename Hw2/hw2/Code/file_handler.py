import csv

import pandas as pd

path = ("DataSet1.csv")


def load_data():
    data_set = []
    X = []
    Y = []
    dataSet_file = open(path, 'r')
    csv_reader = csv.reader(dataSet_file)
    dataCount = 0
    for row in csv_reader:

        if dataCount != 0:
            X.append(float(row[0].replace("'", "")))
            Y.append(float(row[1].replace("'", "")))
            new_element = {'x': row[0], 'y': row[1]}
            data_set.append(new_element)
        dataCount = dataCount + 1

    result = list()
    result.append(X)
    result.append(Y)

    # plt.plot(X,Y)
    # plt.show()
    return result


def read_from_file(filename):
    """
    read data points from csv
    :return: array of data
    """
    data = pd.read_csv(filename, header=None)
    X = data.iloc[:, 0]
    Y = data.iloc[:, 1]
    # print(data[0])
    return data


def prepare_data(path):
    x0 = []
    y0 = []
    x1 = []
    y1 = []

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
    # plt.scatter(x0, y0, color="red")
    # plt.scatter(x1, y1, color="green")
    # plt.title('data')
    # plt.show()
    return dataset_values


"""  testsite_array = []
    with open(filename) as my_file:
        for line in my_file:
            testsite_array.append(line)
    return testsite_array

"""
#
# l=read_from_file('Dataset1.csv')
# print(l[0])
