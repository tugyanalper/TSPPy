class Coordinate(object):
  
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return "Coordinate : x = %d, y = %d" % (self.x, self.y)


def wrapper(func):
    def checker(a, b):
        if a.x < 0 or a.y < 0:
            a = Coordinate(a.x if a.x > 0 else 0, a.y if a.y > 0 else 0)
    
        if b.x < 0 or b.y < 0:
            b = Coordinate(b.x if b.x > 0 else 0, b.y if b.y > 0 else 0)
    
        ret = func(a, b)
    
        if ret.x < 0 or ret.y < 0:
            ret = Coordinate(ret.x if ret.x > 0 else 0, ret.y if ret.y > 0 else 0)
    
        return ret
    return checker


@wrapper      
def add(a, b):
    return Coordinate(a.x + b.x, a.y + b.y)


@wrapper
def sub(a, b):
    return Coordinate(a.x - b.x, a.y - b.y)
  
C1 = Coordinate(100, 300)
C2 = Coordinate(300, 200)

# add = wrapper(add)  # expilicit decorator generation
# sub  = wrapper(sub) # expilicit decorator generation

print sub(C1, C2)
print add(C1, C2)


def one(*args):
    print args

one()
one(1, 2, 3)


def two(x, y, *args):
    print x, y, args
  
two(1, 2, 3)


def foo(**kwargs):
    print kwargs

foo()
foo(x=1, y=1)


def logger(function):
    def inner(*args, **kwargs):
        print "Arguments were %s, %s" % (args, kwargs)
        print function(*args, **kwargs)
    return inner


@logger
def foo1(x, y=1):
    return x*y


@logger
def foo2():
    return 2

print foo1
foo1(5, 4)
foo1(1)
foo2()
