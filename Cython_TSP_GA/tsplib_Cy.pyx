cdef extern from "math.h":
    double sqrt(double x)
    double sin(double x)
    double cos(double x)
    double acos(double x)
    double ceil(double x)

import numpy as np
cimport numpy as np
cimport cython


DTYPE = np.int
ctypedef np.int_t DTYPE_t


cdef int distL2(c1, c2):
    """Compute the L2-norm (Euclidean) distance between two points.

    The distance is rounded to the closest integer, for compatibility
    with the TSPLIB convention.

    The two points are located on coordinates (x1,y1) and (x2,y2),
    sent as parameters
    x1 = c1[0] , y1=c1[1]
    x2 = c2[0] , y2 = c2[1]
    """
    cdef double xdif
    cdef double ydiff
    xdiff = c2[0] - c1[0]
    ydiff = c2[1] - c1[1]
    return int(sqrt(xdiff * xdiff + ydiff * ydiff) + .5)


cdef int distL1(c1, c2):
    """Compute the L1-norm (Manhattan) distance between two points.

    The distance is rounded to the closest integer, for compatibility
    with the TSPLIB convention.

    The two points are located on coordinates (x1,y1) and (x2,y2),
    sent as parameters"""
    return int(abs(c2[0] - c1[0]) + abs(c2[1] - c1[1]) + .5)


cdef int distLinf(c1, c2):
    """Compute the Linfty distance between two points (see TSPLIB documentation)"""
    return int(max(abs(c2[0] - c1[0]), abs(c2[1] - c1[1])))


def distATT(p1, p2): # (x1,y1), (x2,y2)
    """Compute the ATT distance between two points (see TSPLIB documentation)"""
    cdef int xd, yd 
    cdef float tij, rij
    xd = p2[0] - p1[0]
    yd = p2[1] - p1[1]
    rij = sqrt((xd * xd + yd * yd) / 10.)
    tij = int(rij + .5)
    if tij < rij:
        return tij + 1
    else:
        return tij


cdef distCEIL2D(p1, p2):
    cdef float xdiff, ydiff
    xdiff = p2[0] - p1[0]
    ydiff = p2[1] - p1[1]
    return int(ceil(sqrt(xdiff * xdiff + ydiff * ydiff)))


cdef distGEO(p1, p2):
    # print "Implementation is wrong"
    # assert False

    cdef float PI = 3.141592, RRR = 6378.388
    cdef int deg
    cdef float lat1, lat2, long1, long2, min_, q1, q2, q3

    deg = int(p1[0] + .5)
    min_ = p1[0] - deg
    lat1 = PI * (deg + 5. * min_ / 3) / 180.
    deg = int(p1[1] + .5)
    min_ = p1[1] - deg
    long1 = PI * (deg + 5. * min_ / 3) / 180.
    deg = int(p2[0] + .5)
    min_ = p2[0] - deg
    lat2 = PI * (deg + 5. * min_ / 3) / 180.
    deg = int(p2[1] + .5)
    min_ = p2[1] - deg
    long2 = PI * (deg + 5. * min_ / 3) / 180.

    
    q1 = cos(long1 - long2)
    q2 = cos(lat1 - lat2)
    q3 = cos(lat1 + lat2)
    return int(RRR * acos(.5 * ((1. + q1) * q2 - (1. - q1) * q3)) + 1.)


def read_explicit_lowerdiag(f, int n): # f is filename, n is dimension
    c = {}
    cdef int i = 1
    cdef int j = 1
    while True:
        line = f.readline()
        for data in line.split():
            c[j][i] = int(data)
            j += 1
            if j > i:
                i += 1
                j = 1
            if i > n:
                DMat = convert2Dmat(n, c)
                return n, DMat, None, None


def read_explicit_upper(f, int n):
    c = {}
    cdef int i = 1
    cdef int j = 2
    while True:
        line = f.readline()
        for data in line.split():
            c[i, j] = int(data)
            j += 1
            if j > n:
                i += 1
                j = i + 1
            if i == n:
                DMat = convert2Dmat(n, c)
                return n, DMat, None, None


def read_explicit_upperdiag(f, int n):
    c = {}
    cdef int i = 1
    cdef int j = 1
    while True:
        line = f.readline()
        for data in line.split():
            c[i, j] = int(data)
            j += 1
            if j > n:
                i += 1
                j = i
            if i == n:
                DMat = convert2Dmat(n, c)
                return n, DMat, None, None


def read_explicit_matrix(f, int n):
    c = {}
    cdef int i = 1
    cdef int j = 1
    while True:
        line = f.readline()
        for data in line.split():
            if j > i:
                c[i, j] = int(data)
            j += 1
            if j > n:
                i += 1
                j = 1
            if i == n:
                # return range(1, n + 1), c, None, None
                DMat = convert2Dmat(n, c)
                return n, DMat, None, None


# def mk_matrix(coord, dist):
#     """Compute a distance matrix for a set of points.

#     Uses function 'dist' to calculate distance between
#     any two points.  Parameters:
#     -coord -- list of tuples with coordinates of all points, [(x1,y1),...,(xn,yn)]
#     -dist -- distance function
#     """
#     n = len(coord)
#     D = {}  # dictionary to hold n times n matrix
#     for i in range(n - 1):
#         for j in range(i + 1, n):
#             (x1, y1) = coord[i]
#             (x2, y2) = coord[j]
#             D[i, j] = dist((x1, y1), (x2, y2))
#             D[j, i] = D[i, j]
#     return n, D


def convert2Dmat(n, c):
    cdef np.ndarray DMat = np.empty((n,n), dtype= np.int)
    V = np.arange(n, dtype = np.int)
    cdef int ii, jj
    for ii in V:
        for jj in V:
            if jj > ii:
                DMat[ii] [jj] = c[ii+1, jj+1]
            elif jj < ii:
                DMat[ii] [jj] = c[jj+1, ii+1]
            else:
                DMat[ii] [jj] = 0
    return DMat




def read_tsplib(filename):
    filename = '/Users/alperaytun/PycharmProjects/TSP/TSPLIB_Decompressed/' + filename + '.tsp'
    cdef int dimension

    with open(filename, 'r') as f:
        line = f.readline()
        x, y = {}, {}
        while line:
            if line.startswith('DIMENSION'):
                dimension = int(line.split(':')[1])

            elif line.startswith('EDGE_WEIGHT_TYPE'):
                wtype = line.split(':')[1].strip()

                if wtype == 'EUC_2D':
                    dist = distL2
                elif wtype == 'MAN_2D':
                    dist = distL1
                elif wtype == 'MAX_2D':
                    dist = distLinf
                elif wtype == 'ATT':
                    dist = distATT
                elif wtype == 'CEIL_2D':
                    dist = distCEIL2D
                elif wtype == 'GEO':
                    dist = distGEO
                elif wtype == 'EXPLICIT':
                    while line:
                        line = f.readline()
                        if line.startswith('EDGE_WEIGHT_FORMAT'):
                            shape = line.split(':')[1].strip()

                        elif line.startswith('EDGE_WEIGHT_SECTION'):
                            if shape == 'UPPER_ROW':
                                return read_explicit_upper(f, dimension)
                            elif shape == 'LOWER_DIAG_ROW':
                                return read_explicit_lowerdiag(f, dimension)
                            elif shape == 'UPPER_DIAG_ROW':
                                return read_explicit_upperdiag(f, dimension)
                            elif shape == 'FULL_MATRIX':
                                return read_explicit_matrix(f, dimension)
                            else:
                                pass
                        else:
                            pass
                else:
                    print 'cannot deal with {} distances'.format(wtype)
                    raise Exception

            elif line.startswith('NODE_COORD_SECTION'):
                while line:
                    line = f.readline()

                    if line.startswith('EOF'):
                        break
                    else:
                        node_number, xpos, ypos = line.split()
                        x[int(node_number)] = float(xpos)
                        y[int(node_number)] = float(ypos)

            line = f.readline()

    c = {(i, j): dist((x[i], y[i]), (x[j], y[j])) for i in x.keys() for j in y.keys()}
    DMat = convert2Dmat(dimension, c)
    return len(x), DMat, x, y


# def mk_closest(D, n):
#     """Compute a sorted list of the distances for each of the nodes.

#     For each node, the entry is in the form [(d1,i1), (d2,i2), ...]
#     where each tuple is a pair (distance,node).
#     """
#     C = []
#     for i in range(n):
#         dlist = [(D[i, j], j) for j in range(n) if j != i]
#         dlist.sort()
#         C.append(dlist)
#     return C


def length(np.ndarray[DTYPE_t, ndim=1] tour, np.ndarray[DTYPE_t, ndim=2] D):
    """Calculate the length of a tour according to distance matrix 'D'."""
    cdef int z
   
    z = D[tour[-1]] [tour[0]]
    for i in range(1, len(tour)):
        z += D[tour[i]] [tour[i - 1]]
    return z


def best_tour(filename, dMat):
    filename = '/Users/alperaytun/PycharmProjects/TSP/TSPLIB_Decompressed/' + filename + '.opt.tour'
    try:
        f = open(filename, 'r')
    except:
        print "No optimal tour"

    bestTour = []
    line = f.readline()

    while line.find("DIMENSION") == -1:
        line = f.readline()
    [_, dimension] = line.split(":")
    dimension = int(dimension)

    while line.find("TOUR_SECTION") == -1:
        line = f.readline()

    line = f.readline()
    if len(line.split()) > 1:
        bestTour.extend([int(i) for i in line.split()])
    else:
        bestTour.append(int(line))
        for kk in range(1, dimension):
            data = f.readline()
            bestTour.append(int(data))
    print(bestTour)
    
    optimal_length = length(np.array(bestTour)-1, dMat)
    return bestTour, optimal_length
