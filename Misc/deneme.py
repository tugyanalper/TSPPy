import random
from collections import deque


class Employee:
    """Common base class for employees"""

    employeeCount = 0  # This is a class variable

    def __init__(self, name, salary):  # Class constructor
        self.name = name
        self.salary = salary
        Employee.employeeCount += 1

    def display_count(self):
        print "Total Employee %d " % Employee.employeeCount

    def display_employee(self):
        print "Name :", self.name, "Salary :", self.salary

    def __str__(self):
        return "Name : %s, Salary : %d" % (self.name, self.salary)


class tour:
    """ Generates a tour """

    def __init__(self, numOfCities):
        random.randint(1, 2)


class Pet(object):
    """Creates a Pet Object"""

    def __init__(self, name, species):
        self.name = name
        self.species = species

    def getname(self):
        return self.name

    def getspecies(self):
        return self.species

    def __str__(self):
        return "%s is a %s" % (self.name, self.species)


class Dog(Pet):
    """Creates a Dog Object"""

    def __init__(self, name, chases_cats):
        Pet.__init__(self, name, "Dog")
        self.chases_cats = chases_cats

    def chases_cats(self):
        return self.chases_cats


class Cat(Pet):
    def __init__(self, name, hates_dogs):
        Pet.__init__(self, name, "Cat")
        self.hates_dogs = hates_dogs

    def hates_dogs(self):
        return self.hates_dogs


def main():
    list1 = [1, 2, 3, 4]
    list2 = [5, 6, 7, 8]

    list3 = (list1 + list2) * 4
    print list3
    dict1 = {"Name": "Alper", "Age": 35, "TC_No": 64156015738}

    for element in list3:
        print element

    print dict1.keys()
    print dict1.values()
    print "Length of the List3 is ", len(list3)
    print "Length of the Dict1 is ", len(dict1)
    list1.extend(list2)
    print list1
    employee1 = Employee("Alper", 5000)
    employee1.display_employee()
    print employee1.__doc__
    print employee1.__class__
    print employee1.__dict__
    print employee1.__module__
    print employee1

    bonnie = Dog("Bonnie", True)
    minnie = Cat("Minnie", False)

    print bonnie
    print minnie
    print "%s chases cats : %s" % (bonnie.getname(), bonnie.chases_cats)
    print "%s hates dogs : %s" % (minnie.getname(), minnie.hates_dogs)

    d = deque([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    d.rotate(3)
    
    checkPoints = range(0,2001,50)
    print checkPoints


if __name__ == "__main__":
    main()
