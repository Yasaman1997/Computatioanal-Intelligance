from matplotlib import pyplot


def read_data(path):
    dataset_file = open(path, 'r')
    dataset = dataset_file.read()
    # dataset = csv.reader(dataset_file)
    dataset_values = []
    label = []
    x0 = []
    y0 = []
    x1 = []
    y1 = []
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

    dataset_file.close()
    pyplot.scatter(x0, y0, color="red")
    pyplot.scatter(x1, y1, color="green")
    # pyplot.show()
    return dataset_values
