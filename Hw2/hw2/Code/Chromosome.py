import statistics

import numpy


class Chromosome:
    def __init__(self, chromosome_length, min, max, X_dim, Y_dim):
        # Todo create a random list for genes between min and max below
        # Create a ,b and  for every x,y apply them to this a,b
        # every gen contain ØŸa or b valu
        # every chromosome is  (a,b) for data set that reduce dim to 1 .
        self.gene = []
        self.Z = []
        self.score = 0
        a = numpy.random.normal(0, (max - min))
        b = numpy.random.normal(0, (max - min))
        self.gene.append(a / ((a) ** 2 + (b) ** 2) ** 0.5)
        self.gene.append(b / ((a) ** 2 + (b) ** 2) ** 0.5)
        self.gene.append(numpy.random.normal(0.025, 0.05))
        self.X_dim = X_dim
        self.Y_dim = Y_dim
        self.evaluate()

    def evaluate(self):
        # for evaluate method we calculate Zi for every data(Xi,Yi) then get standard derivation of Z for score of chromosome
        self.calculate_1dim_data();
        st_derivation = statistics.stdev(self.Z, statistics.mean(self.Z))
        self.score = st_derivation

    def evluate_update(self):
        self.evaluate()

    def calculate_1dim_data(self):
        i = 0
        for x, y in zip(self.X_dim, self.Y_dim):
            zi = self.gene[0] * x + self.gene[1] * y
            self.Z.append(zi)
            i += 1

