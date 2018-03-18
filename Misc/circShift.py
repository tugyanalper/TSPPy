import numpy as np
from numba import jit, int16, int32

@jit (int32(int16, int16))
def multiply_I16(a ,b):
	return a * b


def main():
	for i in xrange(10000):
		print multiply_I16(i, 5)


if __name__ == '__main__':
	main()