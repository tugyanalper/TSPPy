from collections import deque
from time import sleep
from deap import base
from deap import creator
from deap import tools
from numba import jit

import roll
import editdistance
import numpy
import random
import sys
import tsplib

interpreter = sys.executable
intr_str = interpreter.split("/")

# check if the interpreter is pypy
if intr_str[-1] != "pypy":
    from matplotlib import pyplot as plt

# Differential evolution parameters
H = 2  # Upper Limit
L = -2  # Lower Limit

CR = 0.25  # crossover rate
F = 0.05  # scaling factor
NPOP = 100  # population size
NGEN = 1000  # number of generations to execute

# read the tsp file and calculate distance matrix for problem instance
problemname = 'bayg29'
[NDIM, dMat, xpos, ypos] = tsplib.read_tsplib(problemname)
bestsolution, optimalTour = tsplib.best_tour(problemname, dMat)

# define the problem as an Minimization or Maximization by defining the weights
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create a container to hold individuals which have an attribute named "fitness"
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()  # get toolbox class from base module of deap
toolbox.register("randomkey", random.uniform, L, H)  # register attribute method to toolbox
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.randomkey,
                 NDIM)  # register individual method
toolbox.register("population", tools.initRepeat, list, toolbox.individual,
                 NPOP)  # register population method to toolbox
toolbox.register("select", tools.selRandom, k=3)  # select 3 individuals randomly to apply mutation in DE


# User specific functions
# evaluate fitness of each individual
# @jit
def eval_fitness(individual):
    length = tsplib.length(individual, dMat)
    return length,


# apply circular shift to best tour to evaluate edit distance
# @jit(nopython=False)
def circularshift(best):
    temp1 = best
    rotateIdx = (temp1 == bestsolution[0]).nonzero()
    temp1 = numpy.roll(temp1, -rotateIdx[0])
    temp2 = temp1.copy()
    temp2 = numpy.roll(temp2, -1)
    temp2 = numpy.flip(temp2, 0)

    dist1 = editdistance.eval(bestsolution, temp1)
    dist2 = editdistance.eval(bestsolution, temp2)

    if dist1 <= dist2:
        return temp1, dist1
    else:
        return temp2, dist2


# convert real-valued random keys to a tour
def randomkey2tour(individual):
    """
    :param individual: individual is a list of real-valued random key with fitness attribute
    :return: returns a permutation type tour
    """
    tour = sorted(range((len(individual))), key=lambda k: individual[k])
    tour = numpy.array(tour, dtype=numpy.uint16)
    tour = tour + 1
    # tour = [city + 1 for city in tour]
    # temp = sorted(individual)
    # tour = [jj for ii in range(len(individual)) for jj in range(len(temp)) if individual[ii] == temp[jj]]
    return tour


# register objective function, circular shift and randomkey2tour functions
#  with the appropriate aliases to the toolbox
toolbox.register("evaluate", eval_fitness)
toolbox.register("circshift", circularshift)
toolbox.register("rk2t", randomkey2tour)


def main():
    pop = toolbox.population()  # create population
    hof = tools.HallOfFame(1)  # keep the best value in Hall of Fame

    # register statistics to logbook
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max", "avg", "std", "distance", "rate", "cbest"  # assign logbook headers

    # Calculate the fitness values of individuals by converting them
    # from randomkeys to a tour
    # permpop = []
    # for element in pop:
    #     tour = toolbox.rk2t(element)  # convert from random key to tour
    #     permpop.append(tour)

    permpop = numpy.empty((NPOP, NDIM), dtype=numpy.uint16)
    for j, element in enumerate(pop):
        tour = toolbox.rk2t(element)
        permpop[j] = tour

    fitnesses = toolbox.map(toolbox.evaluate, permpop)  # calculate fitness values
    for ind, fit in zip(pop, fitnesses):  # assign fitness values to their belonging individual
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), distance=NDIM, rate=0, **record)
    print(logbook.stream)

    hof.update(pop)
    cbestfitness = hof.items[0].fitness.values[0]

    boolidx = numpy.empty((NGEN,), dtype=numpy.bool_)

    for g in xrange(1, NGEN):
        for k, agent in enumerate(pop):
            # a = tools.selBest(pop, 1)[0]
            a, b, c = toolbox.select(pop)  # select 3 mutually exclusive individuals
            y = toolbox.clone(agent)  # make a copy the target individual

            # generate a random integer in the range NDIM to at least mutate one parameter
            index = random.randrange(NDIM)
            for i, value in enumerate(agent):
                if i == index or random.random() < CR:
                    y[i] = a[i] + F * (b[i] - c[i])  # mutate the individual

                    # check if newly created value is within the upper and lower bound
                    if (y[i] < L) or (y[i] > H):
                        y[i] = L + (H - L) * random.random()  # if not then generate a random number in range

            permy = toolbox.rk2t(y)  # convert real valued mutant into a permutation city tour
            y.fitness.values = toolbox.evaluate(permy)  # calculate the fitness value of the mutant

            # check if the mutant has better fitness value than the original
            if y.fitness > agent.fitness:
                pop[k] = y  # if this is the case then replace the original with the mutant

        hof.update(pop)
        if hof.items[0].fitness.values[0] < cbestfitness:
            cbestfitness = hof.items[0].fitness.values[0]
            boolidx[g] = True
        else:
            boolidx[g] = False

        record = stats.compile(pop)
        bestInPop = randomkey2tour(hof[0])
        bestInPop, dist = toolbox.circshift(bestInPop)

        rate = float(dist) / float(NDIM)
        logbook.record(gen=g, evals=len(pop), distance=dist, rate=rate, **record)
        print(logbook.stream)
        # sleep(0.001)

    # print("Best individual is ", hof[0], hof[0].fitness.values[0])
    besttour = toolbox.rk2t(hof.items[0])
    besttour, distance = circularshift(besttour)

    print("Best tour is", list(besttour), int(cbestfitness))
    print("Best sol  is", bestsolution, optimalTour)
    print ("Edit Distance is", int(distance))

    gen, avg, minval, std, Levdist, Levrate = logbook.select("gen", "avg", "min", "std", "distance", "rate")
    minarray = numpy.array(minval, numpy.int16)
    progressArray = minarray[boolidx]
    print progressArray

    if intr_str[-1] != "pypy":
        plt.subplot(2, 1, 1)
        plt.plot(gen, minval, 'bo')
        plt.xlabel("Number of Generations")
        plt.ylabel("Minimum Fitness Value")

        plt.subplot(2, 1, 2)
        plt.plot(gen, std)
        plt.xlabel("Number of Generations")
        plt.ylabel("Standard Deviation of Fitness Values")

        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.plot(gen, Levdist)
        plt.xlabel("Number of Generations")
        plt.ylabel("Levenstehein Distance of Best Solution")

        plt.subplot(2, 1, 2)
        plt.plot(gen, Levrate)
        plt.xlabel("Number of Generations")
        plt.ylabel("Levenstehein rate of Best Solution")
        plt.show()


if __name__ == "__main__":
    main()
