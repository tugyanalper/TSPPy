from __future__ import print_function, division
from deap import base
from deap import tools
from deap import creator

import random
import tsplib
import numpy as np
from matplotlib import pyplot as plt

CR = 0.1  # crossover rate
F = 1  # scaling factor
NPOP = 50  # population size
NGEN = 1500  # number of generations to execute

# read the tsp file and calculate distance matrix for problem instance
problemname = 'gr24'
[NDIM, dMat, xpos, ypos] = tsplib.read_tsplib2(problemname)
bestsolution, optimalTour = tsplib.best_tour(problemname, dMat)

# define the problem as an Minimization or Maximization by defining the weights
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# create a container to hold individuals which have an attribute named "fitness"
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()  # get toolbox class from base module of deap
toolbox.register("attribute", random.sample, range(1, NDIM + 1), NDIM)  # register attribute method to toolbox
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)  # register individual method
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # register population method to toolbox
toolbox.register("select", tools.selRandom, k=3)  # select 3 individuals randomly to apply mutation in DE


# User specific functions - evaluate fitness of each individual
def eval_fitness(individual):
    """
    
    :param individual: 
    :return: 
    """
    length = tsplib.length(individual, dMat)
    return length,


# forward transformation- from integer to real value
def forward_transformation(val):
    rvi = -1 + ((val * 500) / 999.0)
    return rvi


# backward transformation- from real value to integer
def backward_transformation(rvIndividual):
    tour = []
    for element in rvIndividual:
        temp = ((1 + element) * 999.0) / 500.0
        tour.append(round(temp))
    return map(int, tour)


def isfeasible(individual):
    fInRange = True
    for num in individual:
        if num not in range(1, NDIM + 1):
            fInRange = False

    fUnique = True
    if len(set(individual)) != NDIM:
        fUnique = False
    return fInRange and fUnique


def repair(individual):
    # check first if individuals are in bounds
    for idx, num in enumerate(individual):
        if num < 1:
            individual[idx] = 1
        elif num > NDIM:
            individual[idx] = NDIM

    ind_array = np.array(individual)
    idx_sort = np.argsort(ind_array)
    sorted_ind_array = ind_array[idx_sort]
    vals, idx_start, count = np.unique(sorted_ind_array, return_counts=True, return_index=True)
    res = np.split(idx_sort, idx_start[1:])
    vals = vals[count > 1]
    res = filter(lambda x: x.size > 1, res)
    insertion_index_array = []
    for subarray in res:
        insertion_index_array.extend(subarray[:-1])

    missing_values = set(range(1, NDIM + 1)).difference(set(individual))

    insertion_random_array = random.sample(range(len(missing_values)), len(missing_values))

    for idx, val in enumerate(insertion_random_array):
        individual[insertion_index_array[val]] = list(missing_values)[idx]
    return individual


# register objective function, circular shift and randomkey2tour functions
#  with the appropriate aliases to the toolbox
toolbox.register("evaluate", eval_fitness)
toolbox.register("fwd_t", forward_transformation)
toolbox.register("bck_t", backward_transformation)


def main():
    # random.seed(2003)
    pop = toolbox.population(n=NPOP)  # create population
    hof = tools.HallOfFame(1)  # keep the best value in Hall of Fame

    # register statistics to logbook
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max", "avg", "std"

    fitnesses = toolbox.map(toolbox.evaluate, pop)  # calculate fitness values
    # assign fitness values to their corresponding individuals
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)

    hof.update(pop)  # update hall of fame with new population

    for g in xrange(1, NGEN):

        for k, agent in enumerate(pop):
            a, b, c = toolbox.select(pop)  # select 3 mutually exclusive individuals
            y = toolbox.clone(agent)  # make a copy the target individual
            del y.fitness.values

            # generate a random integer in the range NDIM to at least mutate one parameter
            index = random.randrange(NDIM)

            for i, value in enumerate(agent):  # iterate over the target individual 
                if i == index or random.random() < CR:
                    y[i] = toolbox.fwd_t(a[i]) + F * (
                        toolbox.fwd_t(b[i]) - toolbox.fwd_t(c[i]))  # mutate the individual
                else:
                    y[i] = toolbox.fwd_t(y[i])

            py = creator.Individual(toolbox.bck_t(y))  # convert real valued mutant into a permutation city tour

            # check if mutant is feasible
            if not isfeasible(py):
                py = repair(py)

            py.fitness.values = toolbox.evaluate(py)  # calculate the fitness value of the mutant

            #  check if the mutant has better fitness value than the original
            if py.fitness > agent.fitness:
                    pop[k] = py  # if this is the case then replace the original with the mutant

        hof.update(pop)  # update hall of fame with new population

        record = stats.compile(pop)  # compile statistics with new 
        logbook.record(gen=g, evals=len(pop), **record)  # record to logbook 
        print(logbook.stream)  # print statistics

    print("Best tour is", hof[0], hof[0].fitness.values[0])
    print("Opt. sol  is", bestsolution, optimalTour)

if __name__ == '__main__':
    main()
