import math
import numpy as np
# from numba import jit, int16

def distL2(c1, c2):
    """Compute the L2-norm (Euclidean) distance between two points.

    The distance is rounded to the closest integer, for compatibility
    with the TSPLIB convention.

    The two points are located on coordinates (x1,y1) and (x2,y2),
    sent as parameters
    x1 = c1[0] , y1=c1[1]
    x2 = c2[0] , y2 = c2[1]
    """
    xdiff = c2[0] - c1[0]
    ydiff = c2[1] - c1[1]
    return int(math.sqrt(xdiff * xdiff + ydiff * ydiff) + .5)


def distL1(c1, c2):
    """Compute the L1-norm (Manhattan) distance between two points.

    The distance is rounded to the closest integer, for compatibility
    with the TSPLIB convention.

    The two points are located on coordinates (x1,y1) and (x2,y2),
    sent as parameters"""
    return int(abs(c2[0] - c1[0]) + abs(c2[1] - c1[1]) + .5)


def distLinf(c1, c2):
    """Compute the Linfty distance between two points (see TSPLIB documentation)"""
    return int(max(abs(c2[0] - c1[0]), abs(c2[1] - c1[1])))


def distATT((x1, y1), (x2, y2)):
    """Compute the ATT distance between two points (see TSPLIB documentation)"""
    xd = x2 - x1
    yd = y2 - y1
    rij = math.sqrt((xd * xd + yd * yd) / 10.)
    tij = int(rij + .5)
    if tij < rij:
        return tij + 1
    else:
        return tij


def distCEIL2D((x1, y1), (x2, y2)):
    xdiff = x2 - x1
    ydiff = y2 - y1
    return int(math.ceil(math.sqrt(xdiff * xdiff + ydiff * ydiff)))


def distGEO((x1, y1), (x2, y2)):
    # print "Implementation is wrong"
    # assert False
    PI = 3.141592
    deg = int(x1 + .5)
    min_ = x1 - deg
    lat1 = PI * (deg + 5. * min_ / 3) / 180.
    deg = int(y1 + .5)
    min_ = y1 - deg
    long1 = PI * (deg + 5. * min_ / 3) / 180.
    deg = int(x2 + .5)
    min_ = x2 - deg
    lat2 = PI * (deg + 5. * min_ / 3) / 180.
    deg = int(y2 + .5)
    min_ = y2 - deg
    long2 = PI * (deg + 5. * min_ / 3) / 180.

    RRR = 6378.388
    q1 = math.cos(long1 - long2);
    q2 = math.cos(lat1 - lat2);
    q3 = math.cos(lat1 + lat2);
    return int(RRR * math.acos(.5 * ((1. + q1) * q2 - (1. - q1) * q3)) + 1.)


def read_explicit_lowerdiag(f, n):
    c = {}
    i, j = 1, 1
    while True:
        line = f.readline()
        for data in line.split():
            c[j, i] = int(data)
            j += 1
            if j > i:
                i += 1
                j = 1
            if i > n:
                DMat = convert2Dmat(range(1, n + 1), c)
                return n, DMat, None, None


def read_explicit_upper(f, n):
    c = {}
    i, j = 1, 2
    while True:
        line = f.readline()
        for data in line.split():
            c[i, j] = int(data)
            j += 1
            if j > n:
                i += 1
                j = i + 1
            if i == n:
                DMat = convert2Dmat(range(1, n + 1), c)
                return n, DMat, None, None


def read_explicit_upperdiag(f, n):
    c = {}
    i, j = 1, 1
    while True:
        line = f.readline()
        for data in line.split():
            c[i, j] = int(data)
            j += 1
            if j > n:
                i += 1
                j = i
            if i == n:
                DMat = convert2Dmat(range(1, n + 1), c)
                return n, DMat, None, None


def read_explicit_matrix(f, n):
    c = {}
    i, j = 1, 1
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
                DMat = convert2Dmat(range(1, n + 1), c)
                return n, DMat, None, None


def mk_matrix(coord, dist):
    """Compute a distance matrix for a set of points.

    Uses function 'dist' to calculate distance between
    any two points.  Parameters:
    -coord -- list of tuples with coordinates of all points, [(x1,y1),...,(xn,yn)]
    -dist -- distance function
    """
    n = len(coord)
    D = {}  # dictionary to hold n times n matrix
    for i in range(n - 1):
        for j in range(i + 1, n):
            (x1, y1) = coord[i]
            (x2, y2) = coord[j]
            D[i, j] = dist((x1, y1), (x2, y2))
            D[j, i] = D[i, j]
    return n, D


def convert2Dmat(V, c):
    DMat = {}
    for ii in V:
        for jj in V:
            if jj > ii:
                DMat[ii, jj] = c[ii, jj]
            elif jj < ii:
                DMat[ii, jj] = c[jj, ii]
            else:
                DMat[ii, jj] = 0
    return DMat


# def read_tsplib(filename):
#     "basic function for reading a TSP problem on the TSPLIB format"
#     "NOTE: only works for 2D euclidean, manhattan and Explicit distances"
#     filename = '/Users/alperaytun/PycharmProjects/TSP/TSPLIB_Decompressed/' + filename + '.tsp'
#     f = open(filename, 'r')

#     line = f.readline()

#     while line.find("DIMENSION") == -1:
#         line = f.readline()

#     [_, dimension] = line.split(":")
#     dimension = int(dimension)

#     while line.find("EDGE_WEIGHT_TYPE") == -1:
#         line = f.readline()

#     if line.find("EUC_2D") != -1:
#         dist = distL2

#     elif line.find("MAN_2D") != -1:
#         dist = distL1

#     elif line.find("MAX_2D") != -1:
#         dist = distLinf
#     elif line.find("ATT") != -1:
#         dist = distATT
#     elif line.find("CEIL_2D") != -1:
#         dist = distCEIL2D
#     elif line.find("GEO") != -1:
#         print "geographic"
#         dist = distGEO
#     elif line.find("EXPLICIT") != -1:
#         while line.find("EDGE_WEIGHT_FORMAT") == -1:
#             line = f.readline()
#         if line.find("LOWER_DIAG_ROW") != -1:
#             while line.find("EDGE_WEIGHT_SECTION") == -1:
#                 line = f.readline()
#             a = read_explicit_lowerdiag(f, dimension)
#             return a
#         if line.find("UPPER_ROW") != -1:
#             while line.find("EDGE_WEIGHT_SECTION") == -1:
#                 line = f.readline()
#             return read_explicit_upper(f, dimension)
#         if line.find("UPPER_DIAG_ROW") != -1:
#             while line.find("EDGE_WEIGHT_SECTION") == -1:
#                 line = f.readline()
#             return read_explicit_upperdiag(f, dimension)
#         if line.find("FULL_MATRIX") != -1:
#             while line.find("EDGE_WEIGHT_SECTION") == -1:
#                 line = f.readline()
#             return read_explicit_matrix(f, dimension)
#     else:
#         print "cannot deal with non-euclidean or non-manhattan distances"
#         raise Exception

#     while line.find("NODE_COORD_SECTION") == -1:
#         line = f.readline()

#     x, y = {}, {}
#     while 1:
#         line = f.readline()
#         if line.find("EOF") != -1 or not line:
#             break
#         (i, xi, yi) = line.split()
#         i = int(i)
#         x[i] = float(xi)
#         y[i] = float(yi)

#     V = x.keys()
#     c = {}  # dictionary to hold n times n matrix
#     for i in V:
#         for j in V:
#             c[i, j] = dist((x[i], y[i]), (x[j], y[j]))
#     f.close()
#     # return len(V), c, x, y

def read_tsplib2(filename):
    filename = '/Users/alperaytun/PycharmProjects/TSP/TSPLIB_Decompressed/' + filename + '.tsp'
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
    # c = np.empty([dimension, dimension],dtype='i2')
    # for i in x.keys():
    #     for j in y.keys():
    #         c[i][j] = dist((x[i], y[i]), (x[j], y[j]))
    return len(x), c, x, y


def mk_closest(D, n):
    """Compute a sorted list of the distances for each of the nodes.

    For each node, the entry is in the form [(d1,i1), (d2,i2), ...]
    where each tuple is a pair (distance,node).
    """
    C = []
    for i in range(n):
        dlist = [(D[i, j], j) for j in range(n) if j != i]
        dlist.sort()
        C.append(dlist)
    return C


def length(tour, D):
    """Calculate the length of a tour according to distance matrix 'D'."""

    # try:
    #     z = D[tour[-1], tour[0]]  # edge from last to first city of the tour
    # except KeyError:
    #     z = D[tour[0], tour[-1]]

    # for i in range(1, len(tour)):
    #     try:
    #         z += D[tour[i], tour[i - 1]]  # add length of edge from city i-1 to i
    #     except KeyError:
    #         z += D[tour[i - 1], tour[i]]  # add length of edge from city i-1 to i
    z = D[tour[-1], tour[0]]
    for i in range(1, len(tour)):
        z += D[tour[i], tour[i - 1]]
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

    optimal_length = length(bestTour, dMat)
    return bestTour, optimal_length
