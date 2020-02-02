import csv
import matplotlib.pyplot as plt

dataset_file = open("data.csv")
dataset = csv.reader(dataset_file)
dataset_label = []

label = []
y0 = []
y1 = []
x0 = []
x1 = []
for row in dataset:
    dataset_label.append([[float(row[0].replace("'", "")), float(row[1].replace("'", ""))], int(row[2].replace("'", ""))])
    if int(row[2].replace("'", "")) == 0:
        y0.append(float(row[1].replace("'", "")))
        x0.append(float(row[0].replace("'", "")))
    else:
        y1.append(float(row[1].replace("'", "")))
        x1.append(float(row[0].replace("'", "")))

plt.scatter(x0, y0, color="red")
plt.scatter(x1, y1, color="green")
plt.show()
