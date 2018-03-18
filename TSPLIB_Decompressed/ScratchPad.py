import math


# convert real-valued random keys to a tour
def randomkey2tour(individual):
    """
    :param population: *population* is a list of *individual* class type
    :return: returns a list of tours
    """

    temp = sorted(range(len(individual)), key=lambda k: individual[k])
    # temp = sorted(individual)
    # tour = (jj for ii in range(len(individual)) for jj in range(len(temp)) if individual[ii] == temp[jj])
    # return list(tour)
    return temp


ind = [0.2, 0.5, 0.1, 0.3, 0.05]
#       0    1    2    3    4
tour = (randomkey2tour(ind))
print tour


def is_prime(number):
    if number > 1:
        if number == 2:
            return True
        if number % 2 == 0:
            return False
        for current in range(3, int(math.sqrt(number) + 1), 2):
            if number % current == 0:
                return False
        return True
    return False


def get_primes(number):
    while True:
        if is_prime(number):
            yield number
        number += 1


def solve_number_10():
    total = 2
    for next_prime in get_primes(3):
        if next_prime < 2000000:
            total += next_prime
        else:
            print total
            return

solve_number_10()

