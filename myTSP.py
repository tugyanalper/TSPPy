import numpy as np

import tsplib

NPOP = 80
CXPB = 0.2
MUTPB = 0.8

# Read Distance Matrix from the file
#  Optimal solutions are : gr17 = 2085, gr24 = 1272, gr120 = 6942
problemname = 'gr24'
[NDIM, dMat, xpos, ypos] = tsplib.read_tsplib2(problemname)
bestsolution, optimalTour = tsplib.best_tour(problemname, dMat)
distMat = np.empty((NDIM, NDIM), dtype=float)
for i in range(NDIM):
    for j in range(NDIM):
        distMat[i][j] = dMat[i + 1, j + 1]
del dMat


class Individual(object):
    def __init__(self, dimension):
        self.dimension = dimension
        self.tour = np.random.permutation(dimension)
        self.fitness = None
        self.calculate_fitness()

    def calculate_fitness(self):
        """Calculate the length of a tour according to distance matrix 'D'."""
        self.fitness = distMat[self.tour[-1]][self.tour[0]]
        for i in range(self.dimension):
            self.fitness += distMat[self.tour[i]][self.tour[i - 1]]


class Population(object):
    def __init__(self, popsize, dimension):
        self.populationSize = popsize
        self.dimension = dimension
        self.population = []
        self.initialize_population()

    def initialize_population(self):
        for i in range(self.populationSize):
            self.population.append(Individual(self.dimension))


def main():
    population = Population(NPOP, NDIM)
    print population


if __name__ == '__main__':
    main()
