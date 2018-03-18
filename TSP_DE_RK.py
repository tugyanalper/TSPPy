from __future__ import print_function, division
from collections import deque
from deap import base
from deap import creator
from deap import tools
from matplotlib import pyplot as plt
from matplotlib.markers import CARETDOWN
import numpy.polynomial.polynomial as poly


import editdistance
import numpy as np
import random
import tsplib


# Differential evolution parameters
H = 2.0  # Upper Limit
L = -2.0  # Lower Limit

CR = 0.1  # crossover rate
F = 0.01 # scaling factor
NPOP = 100  # population size
NGEN = 1500  # number of generations to execute

# read the tsp file and calculate distance matrix for problem instance
problemname = 'bayg29'
[NDIM, dMat, xpos, ypos] = tsplib.read_tsplib2(problemname)
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
def eval_fitness(individual):
    """
    
    :param individual: 
    :return: 
    """
    length = tsplib.length(individual, dMat)
    return length,


# apply circular shift to best tour to evaluate edit distance
def circularshift(best):
    """
    
    :param best: 
    :return: 
    """
    temp1 = best
    rotateIdx = temp1.index(bestsolution[0])

    temp1 = deque(temp1)
    temp1.rotate(-rotateIdx)
    temp2 = deque(temp1)
    temp2.rotate(-1)

    temp1 = list(temp1)
    temp2 = list(reversed(temp2))
    dist1 = editdistance.bycython.eval(bestsolution, temp1)
    dist2 = editdistance.bycython.eval(bestsolution, temp2)

    if dist1 <= dist2:
        return temp1, dist1
    else:
        return temp2, dist2


# convert real-valued random keys to a tour
def randomkey2tour(individual):
    """
    :param individual: individual is a list of real-valued random key 
    with fitness attribute
    :return: returns a permutation type tour
    """
    tour = sorted(xrange(len(individual)), key=lambda k: individual[k])
    tour = [city + 1 for city in tour]
    return tour


def fitnessdiversity(population):
    """
    Calculates and returns the number of unique fitness values 
    """
    fitnessvals = [int(individual.fitness.values[0]) for individual in population]
    fitset = set(fitnessvals)
    return len(fitset)


def solutiondiversity(population):
    solutions = [tuple(individual) for individual in population]
    unique_solution_set = set(solutions)
    return len(unique_solution_set)

# register objective function, circular shift and randomkey2tour functions
#  with the appropriate aliases to the toolbox
toolbox.register("evaluate", eval_fitness)
toolbox.register("circshift", circularshift)
toolbox.register("rk2t", randomkey2tour)


def main():
    # random.seed(2003)
    pop = toolbox.population()  # create population
    hof = tools.HallOfFame(1)  # keep the best value in Hall of Fame

    # register statistics to logbook
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max", "avg", "std", 'fitdiv', 'soldiv','distance'
    
    # Calculate the fitness values of individuals by converting them
    # from randomkeys to a tour
    permpop = [toolbox.rk2t(element) for element in pop]  # convert from random key to tour

    fitnesses = toolbox.map(toolbox.evaluate, permpop)  # calculate fitness values
    # assign fitness values to their corresponding individuals
    for ind, fit in zip(pop, fitnesses):  
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), distance=NDIM, fitdiv=fitnessdiversity(pop),
        soldiv= solutiondiversity(permpop), **record)
    print(logbook.stream)

    hof.update(pop)
    cbestfitness = hof.items[0].fitness.values[0]

    boolidx = np.empty((NGEN,), dtype=np.bool_)  # create an empty bool array to hold progress flags

    for g in xrange(1, NGEN):

        for k, agent in enumerate(pop):
            a, b, c = toolbox.select(pop)  # select 3 mutually exclusive individuals
            y = toolbox.clone(agent)  # make a copy the target individual

            # generate a random integer in the range NDIM to at least mutate one parameter
            index = random.randrange(NDIM)
            for i, value in enumerate(agent):  # iterate over the target individual 
                if i == index or random.random() < CR:
                    y[i] = a[i] + F * (b[i] - c[i])  # mutate the individual

                    # check if newly created value is within the upper and lower bound
                    if y[i] < L or y[i] > H:
                        y[i] = L + (H - L) * random.random()  # if not then generate a random number in range

            permy = toolbox.rk2t(y)  # convert real valued mutant into a permutation city tour
            y.fitness.values = toolbox.evaluate(permy)  # calculate the fitness value of the mutant

            # check if the mutant has better fitness value than the original
            if y.fitness > agent.fitness:
                pop[k] = y  # if this is the case then replace the original with the mutant

        hof.update(pop)  # update hall of fame with new population

        # if the best individual in new population has a better value than 
        # the previous best then update the current best fitness and mark the 
        # progress
        if hof.items[0].fitness.values[0] < cbestfitness:  
            cbestfitness = hof.items[0].fitness.values[0]
            boolidx[g] = True  # mark the generation as a progress
        else:
            boolidx[g] = False  # no progress

        record = stats.compile(pop)  # compile statistics with new 
        bestInPop = randomkey2tour(hof[0])  # convert best individual to permutation tour
        
        # apply circular shift and find the edit distance 
        bestInPop, dist = toolbox.circshift(bestInPop) 

        logbook.record(gen=g, evals=len(pop), distance=dist, fitdiv=fitnessdiversity(pop),
        soldiv= solutiondiversity(pop), **record)  # record to logbook 
        print(logbook.stream)  # print statistics

    
    besttour = toolbox.rk2t(hof[0])  # convert best individual to random key
    besttour, distance = circularshift(besttour)  # apply circular shift and get dist.

    print("Best tour is", list(besttour), hof[0].fitness.values[0])
    print("Opt. sol  is", bestsolution, optimalTour)
    print ("Edit Distance is", int(distance))

    # get records from logbook with their corresponding names
    gen, minval, std, Levdist = logbook.select("gen", "min", "std", "distance")

    progressArray = [element for i, element in enumerate(minval) if boolidx[i] == True]
    progressArray.insert(0, minval[0])  # put first value to the progress array
    print("Progress Array : ", progressArray)
  
    genArray = [0] + [element for i, element in enumerate(range(0,NGEN)) if boolidx[i] == True]
    print("Progress in Gens. : ", genArray)

    distArray = [Levdist[0]]+[int(element) for i, element in enumerate(Levdist) if boolidx[i] == True]
    print('Progress in Distance : ', distArray)

    # convergence = []
    # for idx in range(len(genArray) - 1):
    #     valdiff = progressArray[idx] - progressArray[idx + 1]
    #     gendiff = genArray[idx + 1] - genArray[idx]
    #     conv_rate = float(valdiff) / gendiff
    #     convergence.append(conv_rate)

    convergence = [(progressArray[idx] - progressArray[idx + 1]) / (genArray[idx + 1] - genArray[idx]) for idx in range(len(genArray)-1) ]

    print("Convergence Array : ", convergence)

    z = poly.polyfit(gen, minval, 4)
    ffit = poly.polyval(gen, z)
  

    plt.figure(1)
    # plt.subplot(2, 2, 1)
    plt.plot(gen, minval, 'bo')
    plt.plot(gen, ffit, 'r')
    plt.plot(gen, minval,'k', markevery=genArray, ls="", marker=CARETDOWN, markersize=12, markerfacecolor='g', label="points")
    plt.xlabel("Number of Generations")
    plt.ylabel("Minimum Fitness Value")
    plt.grid()
    plt.show()


    # plt.subplot(2, 2, 3)
    # plt.plot(gen, std)
    # plt.xlabel("Number of Generations")
    # plt.ylabel("Standard Deviation of Fitness Values")

    # plt.figure(2)
    # plt.subplot(2, 2, 2)
    # plt.plot(gen, Levdist)
    # plt.xlabel("Number of Generations")
    # plt.ylabel("Levenstehein Distance of Best Solution")


if __name__ == "__main__":
    main()
