from collections import deque
# from time import sleep
from deap import base
from deap import creator
from deap import tools

import editdistance
import numpy as np
import random
import sys
import tsplib

interpreter = sys.executable
intr_str = interpreter.split("/")

# check if the interpreter is pypy
if intr_str[-1] != "pypy":
    from matplotlib import pyplot as plt

Pc = 0.8  # crossover probability
Pm = 0.2  # perturbation probability
NPOP = 40  # population size
NGEN = 2000  # number of generations to execute

# read the tsp file and calculate distance matrix for problem instance
problemname = 'bayg29'
[NDIM, dMat, xpos, ypos] = tsplib.read_tsplib(problemname)
bestsolution, optimalTour = tsplib.best_tour(problemname, dMat)

# define the problem as an Minimization or Maximization by defining the weights
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create a container to hold individuals which have an attribute named "fitness"
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()  # get toolbox class from base module of deap
toolbox.register("indices", random.sample, range(NDIM), NDIM)  # register attribute method to toolbox
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)  # register individual method
toolbox.register("population", tools.initRepeat, list, toolbox.individual,
                 NPOP)  # register population method to toolbox
toolbox.register("crossover", tools.cxPartialyMatched)


# User specific functions
# evaluate fitness of each individual
def eval_fitness(individual):
    """

    :param individual: 
    :return: 
    """
    # Distance Matrix is a dictionary in the for (1, 43) : 472
    # City numbers start at 1 in Distance Matrix
    y = np.array(toolbox.clone(individual))
    y += 1
    length = tsplib.length(y, dMat)
    return length,


# apply circular shift to best tour to evaluate edit distance
def circularshift(best):
    """

    :param best: 
    :return: 
    """
    temp1 = [i + 1 for i in best]
    rotateIdx = temp1.index(bestsolution[0])

    temp1 = deque(temp1)
    temp1.rotate(-rotateIdx)
    temp2 = deque(temp1)
    temp2.rotate(-1)

    temp1 = list(temp1)
    temp2 = list(reversed(temp2))
    dist1 = editdistance.eval(bestsolution, temp1)
    dist2 = editdistance.eval(bestsolution, temp2)

    if dist1 <= dist2:
        return temp1, dist1
    else:
        return temp2, dist2


def swap(individual):
    copy = toolbox.clone(individual)
    swaplist = random.sample(range(len(copy)), 2)  # sample 2 indices from list
    copy[swaplist[0]], copy[swaplist[1]] = copy[swaplist[1]], copy[swaplist[0]]
    return copy


# register objective function, circular shift and randomkey2tour functions
#  with the appropriate aliases to the toolbox
toolbox.register("evaluate", eval_fitness)
toolbox.register("circshift", circularshift)


def main():
    """

    :return: 
    """
    pop = toolbox.population()  # create population
    hof = tools.HallOfFame(1)  # keep the best value in Hall of Fame

    # register statistics to logbook
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max", "avg", "std"  # assign logbook headers

    fitnesses = toolbox.map(toolbox.evaluate, pop)  # calculate fitness values
    for ind, fit in zip(pop, fitnesses):  # assign fitness values to their belonging individual
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    print(logbook.stream)

    hof.update(pop)
    cbestfitness = hof.items[0].fitness.values[0]

    boolidx = np.empty((NGEN,), dtype=np.bool_)

    for g in range(1, NGEN):
        for k, target in enumerate(pop):
            copytarget = toolbox.clone(target)  # make a copy the best individual

            # Mutation to generate mutant individual v
            # generate a random number and check if it is less than perturbation probability Pm
            r = random.random()
            if r < Pm:
                v = swap(hof.items[0])  # perturb the best individual
            else:
                v = hof.items[0]  # keep the best individual from previous individual

            # Crossover to generate trial individual u
            p = random.random()
            if p < Pc:
                u, _ = toolbox.crossover(copytarget, v)
            else:
                u = v

            u.fitness.values = toolbox.evaluate(u)  # calculate the fitness value of the trial

            # check if the trial individual has better fitness value than the target individual
            if u.fitness > target.fitness:
                pop[k] = u  # if this is the case then replace the original with the trial

        hof.update(pop)

        if hof.items[0].fitness.values[0] < cbestfitness:
            cbestfitness = hof.items[0].fitness.values[0]
            boolidx[g] = True
        else:
            boolidx[g] = False

        record = stats.compile(pop)
        # bestInPop = randomkey2tour(hof[0])
        # _, dist = toolbox.circshift(hof.items[0])

        # rate = float(dist) / float(NDIM)
        logbook.record(gen=g, evals=len(pop), **record)
        print(logbook.stream)
        # sleep(0.001)

    besttour, distance = circularshift(hof.items[0])

    print("Best tour is", list(besttour), int(cbestfitness))
    print("Best sol  is", bestsolution, optimalTour)
    print ("Edit Distance is", int(distance))

    gen, avg, minval, std, Levdist, Levrate = logbook.select("gen", "avg", "min", "std", "distance", "rate")
    minarray = np.array(minval, np.int16)
    progressArray = minarray[boolidx].tolist()
    progressArray.insert(0, minarray[0])  # put first value to the progress array
    # print ("Progress Array : ", progressArray)
    genArray = np.array(range(0, NGEN))
    genArray = genArray[boolidx].tolist()  # put first generation to the generation array
    genArray.insert(0, 0)
    # print ("Generation Array : ", genArray)

    convergence = []
    for idx in range(len(genArray) - 1):
        valdiff = progressArray[idx] - progressArray[idx + 1]
        gendiff = genArray[idx + 1] - genArray[idx]
        # conv_rate = 1 - ((abs(float(valdiff) / float(gendiff))) ** (1 / float(gendiff)))
        # conv_rate = 1 - (abs(float(7542-progressArray[idx + 1])/(7542 - progressArray[idx]))**(1/gendiff))
        conv_rate = float(valdiff) / gendiff
        convergence.append(conv_rate)

        # print ("Convergence Array : ", convergence)

        # if intr_str[-1] != "pypy":
        #     plt.figure(1)
        #     plt.subplot(2, 1, 1)
        #     plt.plot(gen, minval, 'bo')
        #     plt.xlabel("Number of Generations")
        #     plt.ylabel("Minimum Fitness Value")
        #     plt.grid()
        #
        #     plt.subplot(2, 1, 2)
        #     plt.plot(gen, std)
        #     plt.xlabel("Number of Generations")
        #     plt.ylabel("Standard Deviation of Fitness Values")
        #
        #     plt.figure(2)
        #     plt.subplot(2, 1, 1)
        #     plt.plot(gen, Levdist)
        #     plt.xlabel("Number of Generations")
        #     plt.ylabel("Levenstehein Distance of Best Solution")
        #
        #     plt.subplot(2, 1, 2)
        #     plt.plot(gen, Levrate)
        #     plt.xlabel("Number of Generations")
        #     plt.ylabel("Levenstehein rate of Best Solution")
        #     plt.show()


if __name__ == "__main__":
    main()
