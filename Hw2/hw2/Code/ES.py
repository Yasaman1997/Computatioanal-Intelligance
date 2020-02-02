import math
import statistics

import numpy

import hw2.Code.file_handler as fh
from hw2.Code import Chromosome


class ES:
    def __init__(self, min, max, dataSet, initial_number, select_number, p_mutation, p_combination, goal_st_derivation):
        self.initial_number = initial_number
        self.select_number = select_number
        self.p_mutation = p_mutation
        self.p_combination = p_combination
        self.sigma = numpy.random.normal(0.025, 0.05)
        self.dataSet = fh.load_data()
        self.Y_dim = dataSet[1]
        self.X_dim = dataSet[0]
        self.max = max
        self.min = min
        self.initial_population_set = []
        self.select_population_set = []
        self.goal_st_derivation = goal_st_derivation

    def initial_population(self):
        for i in range(0, self.initial_number):
            chrom = Chromosome.Chromosome(2, self.min, self.max, self.X_dim, self.Y_dim)
            self.initial_population_set.append(chrom)

    def mutation(self):
        N01 = numpy.random.normal(0.5, 1, 2)
        # print(N01)
        for chrom in self.select_population_set:
            chrom.gene[2] = chrom.gene[2] * math.exp(-numpy.random.normal(0, 1) * self.sigma)
            chrom.gene[0] += N01[0] * chrom.gene[2]
            chrom.gene[1] += N01[1] * chrom.gene[2]
            norm = ((chrom.gene[0]) ** 2 + (chrom.gene[1]) ** 2) ** 0.5
            chrom.gene[0] /= norm
            chrom.gene[1] /= norm
            self.sigma = numpy.random.normal(0, 0.05);

            chrom.evluate_update()

    def re_combination(self):
        num_re_combination = numpy.round(len(self.select_population_set) * self.p_combination)
        select_index_combination = numpy.random.randint(0, len(self.select_population_set) - 1, int(num_re_combination))
        for i in range(0, len(select_index_combination) - 1):
            # try:
            self.select_population_set[select_index_combination[i]].gene[0] = \
                statistics.mean([self.select_population_set[select_index_combination[i]].gene[0],
                                 self.select_population_set[select_index_combination[i + 1]].gene[0]])
            self.select_population_set[select_index_combination[i]].evluate_update()
            norm = ((self.select_population_set[select_index_combination[i]].gene[0]) ** 2 + (
                self.select_population_set[select_index_combination[i]].gene[1]) ** 2) ** 0.5
            self.select_population_set[select_index_combination[i]].gene[0] /= norm
            self.select_population_set[select_index_combination[i]].gene[1] /= norm

            self.select_population_set[select_index_combination[i + 1]].gene[1] = \
                statistics.mean([self.select_population_set[select_index_combination[i]].gene[1],
                                 self.select_population_set[select_index_combination[i + 1]].gene[1]])
            norm = ((self.select_population_set[select_index_combination[i + 1]].gene[0]) ** 2 + (
                self.select_population_set[select_index_combination[i + 1]].gene[1]) ** 2) ** 0.5
            self.select_population_set[select_index_combination[i + 1]].gene[0] /= norm
            self.select_population_set[select_index_combination[i + 1]].gene[1] /= norm
            i += 1

    def evaluate(self):
        for chrom in self.select_population_set:
            chrom.evluate_update()

    # def updat_chrom_score(self):
    #   for index in self.select_population():
    #       index.evaluate_update()
    def select_parent(self):

        for i in range(1, self.select_number):
            select_index = numpy.random.randint(0, self.initial_number - 1, self.initial_number)
            # print(select_index)
            for index in select_index:
                self.select_population_set.append(self.initial_population_set[index])

    def select_next_population(self):
        q_tornoment = 4
        select_chrom_set = [0, 0, 0, 0]
        #
        chrom_set = self.select_population_set
        chrom_set.extend(self.initial_population_set)
        self.initial_population_set.clear()
        for i in range(0, self.initial_number):
            select_index = numpy.random.randint(0, len(chrom_set) - 1 - i, q_tornoment)
            j = 0
            for index in select_index:
                select_chrom_set[j] = chrom_set[index]
                j += 1
            # select_chrom_set.sort(key=lambda x:x.score ,reverse=False)
            sorted(select_chrom_set, key=lambda x: x.score, reverse=True)
            self.initial_population_set.append(select_chrom_set[0])
            chrom_set.remove(select_chrom_set[0])
        # self.initial_population_set.sort(key=lambda x: x.score, reverse=False)
        self.initial_population_set = sorted(self.initial_population_set, key=lambda x: x.score, reverse=True)
        self.select_population_set.clear()

    def evolution_process(self):

        self.initial_population()
        i = 0
        while True:
            self.select_parent()
            self.re_combination()
            self.evaluate()
            self.select_next_population()
            i += 1
            print("best chromosome:", self.initial_population_set[0].score,
                  statistics.stdev(self.initial_population_set[0].Z, statistics.mean(self.initial_population_set[0].Z)))
            print("worst chromosome:", self.initial_population_set[self.initial_number - 1].score)
            print("mean score:", statistics.mean(chrom.score for chrom in self.initial_population_set))

            print("*********************************************")
            if i > 100:
                break

        return self.initial_population_set[0]


# data = fh.read_from_file('Dataset1.csv')
# for i in range(2, 32, 2):
#     es = ES(0, 1, data, 1, 10, 0.1, 0.5, 100)
#     es.evolution_process()

# es = ES(min, max, dataSet, initial_number, select_number, p_mutation, p_combination, goal_st_derivation)
