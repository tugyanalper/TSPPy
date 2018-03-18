from __future__ import print_function, division
from deap import base
from deap import tools
from deap import creator
from collections import deque
import array

import random
import editdistance
import numpy as np
cimport numpy as np
from matplotlib import pyplot as plt
from matplotlib.markers import CARETDOWN
import numpy.polynomial.polynomial as poly
import tsplib_Cy
cdef extern from "math.h":
    float atan2(float x, float y)
    

cdef float  PI = 3.141592653589793
cdef int NGEN = 2000
cdef int NPOP = 80
cdef float CXPB = 0.8
cdef float MUTPB = 0.2
cdef bint fApplyDiversity = False
cdef bint fPlugOptimumSolution = False
cdef int  NDIV = 70


DTYPE = np.int
ctypedef np.int_t DTYPE_t

# Read Distance Matrix from the file
#  Optimal solutions are : gr17 = 2085, gr24 = 1272, gr120 = 6942
problemname = 'bays29'
[NDIM, dMat, xpos, ypos] = tsplib_Cy.read_tsplib(problemname)
bestsolution, optimalTour = tsplib_Cy.best_tour(problemname, dMat)

# define the problem as an Minimization or Maximization by defining the weights
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create a container to hold individuals which have an attribute named "fitness"
creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMin)

toolbox = base.Toolbox()  # get toolbox class from base module of deap
toolbox.register("attribute", random.sample, range(NDIM), NDIM)  # register attribute method to toolbox
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)  # register individual method
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # register population method to toolbox


# Define Objective Function
def eval_fitness(individual):
    individual = np.array(individual, dtype = np.int)  # add 1 for indexing since dMat has a keys starting with 1
    return tsplib_Cy.length(individual, dMat),  # add 1 for indexing since dMat has a keys starting with 1


def circularshift(np.ndarray[DTYPE_t, ndim=1] best):
   
    temp1 = best
    cdef int rotateIdx
    arr = np.where(temp1 == bestsolution[0])
    rotateIdx = arr[0]
    temp1 = deque(temp1)
    temp1.rotate(-rotateIdx)
    temp2 = deque(temp1)
    temp2.rotate(-1)

    # temp1 = np.array(temp1, dtype=np.int)
    # temp2 = np.array(reversed(temp2), dtype=np.int)
    temp1 = list(temp1)
    temp2 = list(reversed(temp2))

    cdef int dist1, dist2
    dist1 = editdistance.bycython.eval(bestsolution, temp1)
    dist2 = editdistance.bycython.eval(bestsolution, temp2)

    if dist1 <= dist2:
        return temp1, dist1
    else:
        return temp2, dist2


cdef int fitnessdiversity(population):
    """
    Calculates and returns the number of unique fitness values 
    """
    fitnessvals = np.array([individual.fitness.values[0] for individual in population])
    fitset = set(fitnessvals)
    return len(fitset)


def solutiondiversity(population):
    solutions = [tuple(individual) for individual in population]
    unique_solution_set = set(solutions)
    unique_pop = map(creator.Individual, list(unique_solution_set))
    fitnesses = map(toolbox.evaluate, unique_pop)
    for ind, fit in zip(unique_pop, fitnesses):
        ind.fitness.values = fit
    return len(unique_solution_set), unique_pop


cdef inline swap(individual):
    cdef int idx1, idx
    idx1, idx2 = random.sample(range(NDIM), 2)
    cdef int a =  individual.index(idx1)
    cdef int b =  individual.index(idx2)
    individual[b], individual[a] = individual[a], individual[b]
    return individual


cdef inline create_offspring(pop):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))

    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))  # clone offsprings

    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    return offspring


# register objective function with the "evaluate" alias to the toolbox
toolbox.register("evaluate", eval_fitness)

# register crossover function with the "mate" alias to the toolbox
toolbox.register("mate", tools.cxOrdered)

# register mutation function with the "mutate" alias to the toolbox
toolbox.register("mutate", swap)

# register selection function with the "select" alias to the toolbox
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register('circshift', circularshift)


def tsp():
    # random.seed(169)

    if fPlugOptimumSolution:
        pop = toolbox.population(n=NPOP-1)
        optimumSol = creator.Individual(np.array(bestsolution) - 1)
        fitness = toolbox.evaluate(optimumSol) # returns a tuple
        optimumSol.fitness.values = fitness
        pop.append(optimumSol)
    else:
        pop = toolbox.population(n=NPOP)

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    # stats.register("diversity", fitnessdiversity)

    logbook = tools.Logbook()

    # assign logbook headers
    logbook.header = "gen", "evals", "min", "max", "avg", "std",'distance'

    # populasyondaki her bir bireyin degerini hesapla ve
    # o bireyin fitness degerini guncelle
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), distance=NDIM, 
        soldiv=solutiondiversity(pop)[0],fitdiv=fitnessdiversity(pop),**record) 
    print(logbook.stream)
    hof.update(pop)

    cdef int cbestfitness
    cbestfitness = hof.items[0].fitness.values[0]

    boolidx = np.empty((NGEN,), dtype=np.bool_)  # create an empty bool array to hold progress flags

    for gen in xrange(1, NGEN):

        if fApplyDiversity:
            numberOfUniqueSolutions, new_pop = solutiondiversity(pop)
            if numberOfUniqueSolutions < NDIV:
                # print(tools.selBest(new_pop, 1)[0].fitness.values[0])
                while len(new_pop) != NPOP:
                    new_individual = toolbox.individual()
                    if new_individual not in new_pop:
                        fitness = toolbox.evaluate(new_individual)
                        new_individual.fitness.values = fitness
                        new_pop.append(new_individual)
                # offspring = create_offspring(new_pop)
                # pop[:] = tools.selBest(new_pop, 1) + tools.selBest(offspring, NPOP - 1)
                pop[:] = new_pop
            else:
                offspring = create_offspring(pop)
                pop[:] = tools.selBest(pop, 1) + tools.selBest(offspring, NPOP - 1)
        else:
            offspring = create_offspring(pop)
            pop[:] = tools.selBest(pop, 1) + tools.selBest(offspring, NPOP - 1)

        hof.update(pop)
        bestInPop, dist = toolbox.circshift(np.array(hof[0]))

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(pop),distance=dist, 
            soldiv=solutiondiversity(pop)[0], fitdiv=fitnessdiversity(pop),**record) 
        print(logbook.stream) 

        # if the best individual in new population has a better value than
        # the previous best then update the current best fitness and mark the
        # progress
        if hof.items[0].fitness.values[0] < cbestfitness:
            cbestfitness = hof.items[0].fitness.values[0]
            boolidx[gen] = True  # mark the generation as a progress
        else:
            boolidx[gen] = False  # no progress

    print("-- End of (successful) evolution --")

    best_ind, dist = toolbox.circshift(np.array(hof[0]) + 1)

    print("Best {:>}, {:>}".format(best_ind, hof[0].fitness.values[0]))
    # print "Opt. individual is %s, %s" % (optimum_individual, eval_fitness(bInd))
    print('Opt. {:>}, {:>}'.format(bestsolution, optimalTour))
    gen, evals, avg, minval, fitness_diversity, std, solution_diversity, Levdist = logbook.select('gen', 'evals', 'avg', 'min', 'fitdiv', 'std', 'soldiv', 'distance')

    progressArray = [element for i, element in enumerate(minval) if boolidx[i]]
    progressArray.insert(0, minval[0])  # put first value to the progress array
    # print("Progress Array : ", progressArray)

    genArray = [0] + [element for i, element in enumerate(range(0, NGEN)) if boolidx[i]]
    # print("Progress in Gens. : ", genArray)

    # distArray = [Levdist[0]] + [int(element) for i, element in enumerate(Levdist) if boolidx[i]]
    # print('Progress in Distance : ', distArray)

    convergenceDegrees = [atan2((progressArray[idx] - progressArray[idx + 1]) , (genArray[idx + 1] - genArray[idx])) * 180 / PI for idx in
                   range(len(genArray) - 1)]

    # print("Convergence Speed : ", sum(convergence) / len(convergence))

    z = poly.polyfit(gen, minval, 5)
    ffit = poly.polyval(gen, z)

    # plt.figure(1)
    # plt.subplot(2, 2, 1)
    # plt.plot(gen, minval, 'b-')
    # plt.ylabel("Minimum Tour Distance")
    # plt.xlabel("Number of Generations")
    # plt.axis("tight")

    # plt.subplot(2, 2, 2)
    # plt.plot(gen, avg, 'r-')
    # plt.ylabel("Average Tour Distance")
    # plt.xlabel("Number of Generations")
    # plt.axis("tight")

    # plt.subplot(2, 2, 3)
    # plt.plot(gen, fitness_diversity, 'k-')
    # plt.ylabel("Number of Unique Fitness Values")
    # plt.xlabel("Number of Generations")
    # plt.axis("tight")

    # plt.subplot(2, 2, 4)
    # plt.plot(gen, std, 'g-')
    # plt.ylabel("Standard Deviation of Fitness Values")
    # plt.xlabel("Number of Generations")
    # plt.axis("tight")

    plt.figure(2)
    plt.plot(gen, minval, 'bo')
    # plt.plot(gen, ffit, 'r', linewidth=2)
    # plt.plot(gen, minval, 'k', markevery=genArray, ls="", marker='|', markersize=20, markerfacecolor='b',
    #          label="points")
    plt.xlabel("Number of Generations")
    plt.ylabel("Minimum Fitness Value")
    plt.grid()


    increment = 100
    checkPoints = range(0, NGEN, increment)
    checkPoints.append(NGEN-1)
    tangetArray = []
    for idx in range(len(checkPoints) - 1):
        value_difference = minval[checkPoints[idx]] - minval[checkPoints[idx + 1]]
        plt.plot([checkPoints[idx], checkPoints[idx+1]], [minval[checkPoints[idx]],minval[checkPoints[idx+1]]], linewidth=3)
        tanget_degree = atan2(value_difference, increment) * 180 / PI
        tangetArray.append(tanget_degree)
    print('Sum of tanget degrees : ', sum(tangetArray))
    average_tangent_degree = sum(tangetArray) / len(tangetArray)
    print('Avaerage value of tangent degrees : ', average_tangent_degree)

    great_value_difference = minval[checkPoints[0]] - minval[checkPoints[-1]]
    great_tanget_degree = atan2(great_value_difference, NGEN) * 180 / PI
    print('Tangent Degree for whole generation :', great_tanget_degree )
    plt.plot([checkPoints[0], checkPoints[-1]], [minval[checkPoints[0]], minval[checkPoints[-1]]], linewidth=5)
    try:
        convergence_index = average_tangent_degree /  great_tanget_degree
    except ZeroDivisionError:
        convergence_index = 1

    print('Convergence Index', convergence_index)

    print("Buna Bak :", convergenceDegrees)
    print(sum(convergenceDegrees))
    print(len(convergenceDegrees))
    print(sum(convergenceDegrees) / len(convergenceDegrees))
    print(great_tanget_degree)
    print(great_tanget_degree / (sum(convergenceDegrees) / len(convergenceDegrees)))

    plt.show()


if __name__ == "__main__":
    tsp()
