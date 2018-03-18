from __future__ import print_function, division
from deap import base
from deap import tools
from deap import creator
from collections import deque
from numba import jit
import array

import random
import editdistance
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.markers import CARETDOWN
import numpy.polynomial.polynomial as poly
import tsplib
import math

NGEN = 2000
NPOP = 80
CXPB = 0.8
MUTPB = 0.2
fApplyDiversity = False
fPlugOptimumSolution = False
NDIV = 70

# Read Distance Matrix from the file
#  Optimal solutions are : gr17 = 2085, gr24 = 1272, gr120 = 6942
problemname = 'bays29'
[NDIM, dMat, xpos, ypos] = tsplib.read_tsplib2(problemname)
bestsolution, optimalTour = tsplib.best_tour(problemname, dMat)

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
    individual = np.array(individual)+1  # add 1 for indexing since dMat has a keys starting with 1
    return tsplib.length(individual, dMat),  # add 1 for indexing since dMat has a keys starting with 1


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
    unique_pop = map(creator.Individual, list(unique_solution_set))
    fitnesses = map(toolbox.evaluate, unique_pop)
    for ind, fit in zip(unique_pop, fitnesses):
        ind.fitness.values = fit
    return len(unique_solution_set), unique_pop


def swap(individual):
    idx1, idx2 = random.sample(range(NDIM), 2)
    a, b = individual.index(idx1), individual.index(idx2)
    individual[b], individual[a] = individual[a], individual[b]
    return individual


def create_offspring(pop):
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


def main():
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
    logbook.header = "gen", "evals", "min", "max", "avg", "std", 'fitdiv', 'soldiv', 'distance'

    # populasyondaki her bir bireyin degerini hesapla ve
    # o bireyin fitness degerini guncelle
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), fitdiv=fitnessdiversity(pop),
                   soldiv=solutiondiversity(pop)[0], distance=NDIM, **record)
    print(logbook.stream)
    hof.update(pop)

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
        bestInPop, dist = toolbox.circshift([i + 1 for i in hof[0]])

        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(pop), fitdiv=fitnessdiversity(pop),
                       soldiv=solutiondiversity(pop)[0], distance=dist, **record)
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

    best_ind, dist = toolbox.circshift([i + 1 for i in hof[0]])

    print("Best {:>}, {:>}".format(best_ind, hof[0].fitness.values[0]))
    # print "Opt. individual is %s, %s" % (optimum_individual, eval_fitness(bInd))
    print('Opt. {:>}, {:>}'.format(bestsolution, optimalTour))
    gen, evals, avg, minval, fitness_diversity, std, solution_diversity, Levdist = \
        logbook.select('gen', 'evals', 'avg', 'min', 'fitdiv', 'std', 'soldiv', 'distance')

    progressArray = [element for i, element in enumerate(minval) if boolidx[i]]
    progressArray.insert(0, minval[0])  # put first value to the progress array
    # print("Progress Array : ", progressArray)

    genArray = [0] + [element for i, element in enumerate(range(0, NGEN)) if boolidx[i]]
    # print("Progress in Gens. : ", genArray)

    # distArray = [Levdist[0]] + [int(element) for i, element in enumerate(Levdist) if boolidx[i]]
    # print('Progress in Distance : ', distArray)

    convergenceDegrees = [math.atan2((progressArray[idx] - progressArray[idx + 1]) , (genArray[idx + 1] - genArray[idx])) * 180 / math.pi for idx in
                   range(len(genArray) - 1)]

    # print("Convergence Speed : ", sum(convergence) / len(convergence))

    z = poly.polyfit(gen, minval, 5)
    ffit = poly.polyval(gen, z)

    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.plot(gen, minval, 'b-')
    plt.ylabel("Minimum Tour Distance")
    plt.xlabel("Number of Generations")
    plt.axis("tight")

    plt.subplot(2, 2, 2)
    plt.plot(gen, avg, 'r-')
    plt.ylabel("Average Tour Distance")
    plt.xlabel("Number of Generations")
    plt.axis("tight")

    plt.subplot(2, 2, 3)
    plt.plot(gen, fitness_diversity, 'k-')
    plt.ylabel("Number of Unique Fitness Values")
    plt.xlabel("Number of Generations")
    plt.axis("tight")

    plt.subplot(2, 2, 4)
    plt.plot(gen, std, 'g-')
    plt.ylabel("Standard Deviation of Fitness Values")
    plt.xlabel("Number of Generations")
    plt.axis("tight")

    plt.figure(2)
    plt.plot(gen, minval, 'bo')
    # plt.plot(gen, ffit, 'r', linewidth=2)
    # plt.plot(gen, minval, 'k', markevery=genArray, ls="", marker='|', markersize=20, markerfacecolor='b',
    #          label="points")
    plt.xlabel("Number of Generations")
    plt.ylabel("Minimum Fitness Value")
    plt.grid(True)


    # increment = 100
    # checkPoints = range(0, NGEN, increment)
    checkPoints = [0, NGEN-1]
    # checkPoints.append(NGEN-1)
    # tangetArray = []
    # for idx in range(len(checkPoints) - 1):
    #     value_difference = minval[checkPoints[idx]] - minval[checkPoints[idx + 1]]
    #     # plt.plot([checkPoints[idx], checkPoints[idx+1]], [minval[checkPoints[idx]],minval[checkPoints[idx+1]]], linewidth=3)
    #     tanget_degree = math.atan2(value_difference, increment) * 180 / math.pi
    #     tangetArray.append(tanget_degree)
    # print('Sum of tanget degrees : ', sum(tangetArray))
    # average_tangent_degree = sum(tangetArray) / len(tangetArray)
    # print('Avaerage value of tangent degrees : ', average_tangent_degree)

    great_value_difference = minval[checkPoints[0]] - minval[checkPoints[-1]]
    great_tanget_degree = math.atan2(great_value_difference, NGEN) * 180 / math.pi
    # print('Tangent Degree for whole generation :', great_tanget_degree )
    
    # try:
    #     convergence_index = average_tangent_degree /  great_tanget_degree
    # except ZeroDivisionError:
    #     convergence_index = 1

    # print('Convergence Index', convergence_index)



    print("Buna Bak :", convergenceDegrees)
    print(sum(convergenceDegrees))
    print(len(convergenceDegrees))
    avg_conv_degrees = sum(convergenceDegrees) / len(convergenceDegrees)
    print("avg", avg_conv_degrees)
    print("great", great_tanget_degree)
    cIndex = great_tanget_degree / avg_conv_degrees
    print(cIndex)


    print("Area under curve : " , np.trapz([i - minval[NGEN-1] for i in minval], gen))
    print("Total Area : ", NGEN * (minval[0]- minval[NGEN-1]))

    plt.plot([checkPoints[0], checkPoints[-1]], [minval[checkPoints[0]], minval[checkPoints[-1]]], linewidth=5)
    for idx in range(len(genArray) - 1):
        plt.plot([genArray[idx], genArray[idx+1]], [progressArray[idx],
            progressArray[idx+1]], linewidth=3)
    plt.show()

    with open("myfile.txt", "a+") as myfile:
        myfile.write("Crossover Rate : " + str(CXPB))
        myfile.write("\n" + "Mutation Rate : " + str(MUTPB))
        myfile.write("\nMinimum Fitness Values : \n")
        for num in minval:
            myfile.write(str(num) + " ")
        myfile.write("\nConvergence Degrees : \n")
        for num in convergenceDegrees:
            myfile.write(str(num) + " ")
        myfile.write("\nSum of degrees : " + str(sum(convergenceDegrees)))
        myfile.write('\nGreat Tangent Degree : ' + str(great_tanget_degree))
        myfile.write('\nAverage : ' + str(avg_conv_degrees))
        myfile.write("\nConvergence : " + str(cIndex))
        myfile.write("\n" + '-'*50 + '\n' )

if __name__ == "__main__":
    main()
