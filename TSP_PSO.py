import operator
import random
import numpy
import tsplib

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Particle", list, fitness=creator.FitnessMin, speed=list,
               smin=None, smax=None, best=None)

#  Optimal solutions are : gr17 = 2085, gr24 = 1272, gr120 = 6942
problemname ='gr24'
[NDIM, dMat, _, _] = tsplib.read_tsplib2(problemname)


def eval_fitness(individual):
    length = tsplib.length(individual, dMat)
    return length,


def randomkey2tour(part):
    temp = numpy.array(part) 
    permpart = numpy.argsort(-temp) + 1
    return permpart.tolist()


def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part


def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part[:] = list(map(operator.add, part, part.speed))


toolbox = base.Toolbox()
toolbox.register("particle", generate, size=NDIM, pmin=-6, pmax=6, smin=-3, smax=3)
toolbox.register("population", tools.initRepeat, list, toolbox.particle,n=100)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
# toolbox.register("evaluate", benchmarks.h1)
# register objective function with the "evaluate" alias to the toolbox
toolbox.register("evaluate", eval_fitness)


def main():
    pop = toolbox.population()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = 1000
    best = None

    for g in xrange(GEN):
        print "-- Generation %i --" %g
        for part in pop:
            permpart = randomkey2tour(part)
            part.fitness.values = toolbox.evaluate(permpart)

            if not part.best or part.fitness.values < part.best.fitness.values:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values

            if not best or part.fitness.values < best.fitness.values:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values

        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        # logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        # print(logbook.stream)

    bestint = []
    newlist = randomkey2tour(best)
    for element in newlist:
        bestint.append(int(element))

    print bestint, toolbox.evaluate(bestint)
    return pop, logbook, best


if __name__ == "__main__":
    main()
