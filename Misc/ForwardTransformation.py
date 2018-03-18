from deap import base
from deap import tools
from deap import creator

import random
import numpy as np
from matplotlib import pyplot as plt

NGEN = 1500
NDIM = 8
NPOP = 100

# define the problem as an Minimization or Maximization by defining the weights
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create a container to hold individuals which have an attribute named "fitness"
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()  # get toolbox class from base module of deap
toolbox.register("attribute", random.sample, range(NDIM), NDIM)  # register attribute method to toolbox
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)  # register individual method
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # register population method to toolbox


def forward_transformation(individual):
    rvi = [-1 + ((ii * 500) / 999.0) for ii in individual]
    return rvi


def backward_transformation(rvIndividual):
    tour = []
    for element in rvIndividual:
        temp = ((1 + element) * 999.0) / 500.0
        alpha = round(temp + 0.5)
        beta = alpha - temp
        if beta > 0.5:
            temp_star = alpha - 1
        elif beta < 0.5:
            temp_star = alpha
        tour.append(temp_star)
    return map(int, tour)


pop = toolbox.population(n=3)
for ind in pop:
    rvIndividual = forward_transformation(ind)
    print ind
    print rvIndividual

    btTour = backward_transformation(rvIndividual)
    print btTour
