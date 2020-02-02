import numpy as np
from matplotlib import pyplot
import csv
import  math
import  sklearn.utils  as ut

def sigmd(input):
    return (1 / (1 + math.exp(-input)))


#

def compute_y(W, X, b):

    return np.dot(X,W)+ b


def compute_grad(W, X, b, y0, num):
    y =sigmd( compute_y(W, X, b))
    s_g = sigmd(y) * (1 - sigmd(y))
    if num == 0:
        return X[0] * (y - y0) * s_g
    elif num==1:
        return X[1] * (y - y0) * s_g
    else :
    	return  (y - y0) * s_g


def select_train_data():
    print()


def train(dataset_value_label):
    b = np.random.normal(0, 1)
    Ir = 0.01
    W = []
    grad = [0, 0, 0]
    W.append(np.random.normal(0, 1))
    W.append(np.random.normal(0, 1))
    for i in range(0,1000):

        for w in range(0, 3):
            grad[w] = 0
            for data in dataset_value_label:
                # y = compute_y(W, data[0], b)
                grad[w] += compute_grad(W, data[0], b, data[1], w)
        for w in range(0, 2):
              W[w] -= Ir * grad[w]
        b -= Ir* grad[2]
    return [W,b]


def read_data():
    dataset_file = open("data.csv")
    dataset = csv.reader(dataset_file)
    dataset_value_label = []

    label = []
    y0 = []
    y1 = []
    x0 = []
    x1 = []
    for row in dataset:
        dataset_value_label.append([[float(row[0].replace("'", "")), float(row[1].replace("'", ""))], int(row[2].replace("'", ""))])
        if int(row[2].replace("'", "")) == 0:
            y0.append(float(row[1].replace("'", "")))
            x0.append(float(row[0].replace("'", "")))
        else:
            y1.append(float(row[1].replace("'", "")))
            x1.append(float(row[0].replace("'", "")))

#    pyplot.scatter(x0,y0, color="red")
#    pyplot.scatter(x1, y1, color="green")
#    pyplot.show()
    return  dataset_value_label




# def d_sigmd(input) :
# 			return (_math.exp(- input) / (1 - 				math.exp(- input)))
def test(W, b, dataset_value_label_test):
    for data in dataset_value_label_test :
        sigmY = sigmd(np.dot(W,data[0])+b)
        if sigmY > 0.5 :
            data[1] = 1
        else :
            data[1] = 0
    return dataset_value_label_test

def main() :
    y0 = []
    y1 = []
    x0 = []
    x1 = []
    dataset = read_data()
    dataset_value_label_test=[]
    dataset_value_label_train=[]
    dataset_value_label=[]
  #  dataset=np.random.shuffle(dataset)

 #   print(dataset)
    for i in range(0,len(dataset)):
        if i> np.round(0.6 * len(dataset)):

            dataset_value_label_test.append([[dataset[i][0][0], dataset[i][0][1]],dataset[i][1]])
        else:

            dataset_value_label_train.append([[dataset[i][0][0], dataset[i][0][1]],dataset[i][1]])
    W, b = train(dataset_value_label_train)
    # print(dataset_value_label_test)
    dataset_value_label_test = test(W, b, dataset_value_label_test)
    # print(dataset_value_label_test)
    print(dataset)
    for i in range(0,len(dataset_value_label_test)):
        # print(dataset_value_label_test[i][1])
        if dataset_value_label_test[i][1] == 0 :
            y0.append(dataset_value_label_test[i][0][1])
            x0.append(dataset_value_label_test[i][0][0])
        else:
            y1.append(dataset_value_label_test[i][0][1])
            x1.append(dataset_value_label_test[i][0][0])

    pyplot.scatter(x0,y0,color="red")
    pyplot.scatter(x1,y1,color="green")
    pyplot.show()






main()