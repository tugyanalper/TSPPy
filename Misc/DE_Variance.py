from deap import base
from deap import tools
from deap import creator
from time import sleep
import random
import numpy
import math
from matplotlib import pyplot as plt

# Differential evolution parameters
H = 32   # Upper Limit
L = -32  # Lower Limit

CR = 0.5  # crossover rate
F = 0.9  # scaling factor
NPOP = 50  # population size
NGEN = 200  # number of generations to execute
NDIM = 10

# define the problem as an Minimization or Maximization by defining the weights
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create a container to hold individuals which have an attribute named "fitness"
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()  # get toolbox class from base module of deap
toolbox.register("attribute", random.uniform, L, H)  # register attribute method to toolbox
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, NDIM)  # register individual method
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # register population method to toolbox
toolbox.register("select", tools.selRandom, k=3)


def eval_fitness(individual):
    """Ackley Function"""
    firstSum = 0.0
    secondSum = 0.0
    for c in individual:
        firstSum += c**2.0
        secondSum += math.cos(2.0*math.pi*c)
    n = float(len(individual))
    fitness = -20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e
    return fitness,


def variance(pop):
    msum = 0
    for ind in pop:
        msum += ind[0]
    meanPop = msum / float(NPOP)

    vsum = 0
    for ind in pop:
       vsum += ((ind[0]-meanPop)**2)
    var = vsum / float(NPOP)
    return var

# register objective function with the "evaluate" alias to the toolbox
toolbox.register("evaluate", eval_fitness)

toolbox.register("variance", variance)


def main():

    pop = toolbox.population(n=NPOP)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min",  "max", "avg", "std",  "var_aft"

    # Evaluate the individuals
    fitnesses = toolbox.map(toolbox.evaluate,pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    varPop = toolbox.variance(pop)

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(pop), var_aft = varPop, **record)
    print(logbook.stream)

    for g in xrange(1, NGEN):
        # print("-- Generation %i --" % g)

        for k, agent in enumerate(pop):
            # a = tools.selBest(pop, 1)[0]
            a, b, c = toolbox.select(pop)
            y = toolbox.clone(agent)
            index = random.randrange(NDIM)
            for i, value in enumerate(agent):
                if i == index or random.random() < CR:
                    y[i] = a[i] + F * (b[i] - c[i])

            for i, value in enumerate(agent):
                if (y[i] < L) or (y[i] > H):
                    y[i] = L + (H - L) * random.random()

            y.fitness.values = toolbox.evaluate(y)

            if y.fitness > agent.fitness:
                pop[k] = y
        hof.update(pop)
        var_aft = toolbox.variance(pop)
        record = stats.compile(pop)

        logbook.record(gen=g, evals=len(pop), var_aft =var_aft, **record)
        print(logbook.stream)
        sleep(0.02)

    print "Best solution is :", hof
    gen, varianceAfter = logbook.select("gen",  "var_aft")

    plt.plot(gen, varianceAfter)
    plt.show()

if __name__ == "__main__":
    main()

