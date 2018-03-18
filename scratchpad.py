import tsplib
import numpy as np

mylist  = [9,0,27,5,11,8,4,20,23,7,26,22,6,24,15,12,1,25,28,2,19,10,21,13,16,17,14,18,3]
tour = np.array([i+1 for i in  mylist])
print set(tour)

problemname = 'bays29'
[NDIM, dMat, xpos, ypos] = tsplib.read_tsplib2(problemname)

print tsplib.length(tour, dMat)
import math

avg = 50
great = 62

xx = (avg + great) / 2.0
print math.tan(xx * (math.pi / 180))