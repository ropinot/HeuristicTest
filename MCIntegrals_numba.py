from scipy.stats import truncnorm, norm, uniform, beta, triang
from numpy import ndarray, min
from numba import jit
from math import trunc
import multiprocessing as mp

parameters = {'Q1': 38.0,
              'Q2': 67.0,
              'Q3': 25.0,
              'Q4': -1.0,
              'Q5': -1.0,
              'Q6': -1.0,
              'Q7': -1.0,
              'Q8': -1.0,
              'Q9': -1.0,
              'mu1': 30.0,
              'mu2': 70.0,
              'mu3': 15.0,
              'mu4': 0.0,
              'mu5': 0.0,
              'mu6': 0.0,
              'mu7': 0.0,
              'mu8': 0.0,
              'mu9': 0.0,
              'sigma1': 15.0,
              'sigma2': 35.0,
              'sigma3': 7.5,
              'sigma4': 0.0,
              'sigma5': 0.0,
              'sigma6': 0.0,
              'sigma7': 0.0,
              'sigma8': 0.0,
              'sigma9': 0.0,
              'min_intrv1': 0.,
              'min_intrv2': 0.,
              'min_intrv3': 0.,
              'min_intrv4': 0.,
              'min_intrv5': 0.,
              'min_intrv6': 0.,
              'min_intrv7': 0.,
              'min_intrv8': 0.,
              'min_intrv9': 0.,
              'max_intrv1': 100.,
              'max_intrv2': 100.,
              'max_intrv3': 100.,
              'max_intrv4': 1000.,
              'max_intrv5': 1000.,
              'max_intrv6': 1000.,
              'max_intrv7': 1000.,
              'max_intrv8': 1000.,
              'max_intrv9': 1000.,
              'target': 115.,
              'alpha': 0.5,
              'A': 130.,
              'N': 200000,
              'scaling': True,
              }

@jit(nopython=True)
def MIN(a, b):
    return a if a <= b else b

@jit(nopython=True)
def MAX(a, b):
    return a if a >= b else b

@jit(nopython=True)
def ABS(a):
    return a if a > 0.0 else -a


def integral(parameters):
    N = parameters['N']
    target = parameters['target']
    R = parameters['retailers']

    # rv1, rv2, rv3 = ndarray(shape = (N,), dtype=float), ndarray(shape = (N,), dtype=float), ndarray(shape = (N,), dtype=float)
    if not parameters['distribution']:
        print 'No distribution set...abort'
        exit(1)
    # elif parameters['distribution'] == 'truncnorm':
    #     a1, b1 = (parameters['min_intrv1'] - parameters['mu1']) / parameters['sigma1'], (parameters['max_intrv1'] - parameters['mu1']) / parameters['sigma1']
    #     a2, b2 = (parameters['min_intrv2'] - parameters['mu2']) / parameters['sigma2'], (parameters['max_intrv2'] - parameters['mu2']) / parameters['sigma2']
    #     a3, b3 = (parameters['min_intrv3'] - parameters['mu3']) / parameters['sigma3'], (parameters['max_intrv3'] - parameters['mu3']) / parameters['sigma3']
    #     rv1 = truncnorm(a1, b1, loc=parameters['mu1'], scale=parameters['sigma1']).rvs(N)
    #     rv2 = truncnorm(a2, b2, loc=parameters['mu2'], scale=parameters['sigma2']).rvs(N)
    #     rv3 = truncnorm(a3, b3, loc=parameters['mu3'], scale=parameters['sigma3']).rvs(N)
    elif parameters['distribution'] == 'norm':
        rvs = [norm(parameters['mu{}'.format(i)],
                    parameters['sigma{}'.format(i)]).rvs(N) for i in xrange(1, R+1)]

    elif parameters['distribution'] == 'uniform':
        rvs = [uniform(loc=parameters['mu{}'.format(i)],
                       scale=parameters['sigma{}'.format(i)]).rvs(N) for i in xrange(1, R+1)]

    elif parameters['distribution'] == 'triang':
        rvs = [triang(loc=parameters['min_intrv{}'.format(i)],
                      scale=parameters['max_intrv{}'.format(i)],
                      c=parameters['mu{}'.format(i)]).rvs(N) for i in xrange(1, R+1)]

    else:
        print 'Distribution not recognized...abort'
        exit(1)

    if parameters['scaling']:
        #scale the values of Qs in the allowed range such that sum(Q_i) = A
        r = sum([ABS(parameters['Q{}'.format(i)]) for i in xrange(1, R+1)])
        if r == 0.0:
            r = 1.

        # rounding the values, the sum could exceed A
        tot_other_Q = 0.0
        #  set the first R-1 varibles
        for i in xrange(1, R):
            parameters['Q{}'.format(i)] = ABS(parameters['Q{}'.format(i)]) * parameters['A'] / r
            tot_other_Q += parameters['Q{}'.format(i)]

        #  set the R-th variable by difference
        parameters['Q{}'.format(R)] = parameters['A'] - tot_other_Q

    if parameters['retailers'] == 3:
        return _integral3(rvs[0], rvs[1], rvs[2],
                          parameters['Q1'], parameters['Q2'], parameters['Q3'], target)
    elif parameters['retailers'] == 6:
        return _integral6(rvs[0], rvs[1], rvs[2],
                          rvs[3], rvs[4], rvs[5],
                          parameters['Q1'], parameters['Q2'], parameters['Q3'],
                          parameters['Q4'], parameters['Q5'], parameters['Q6'], target)
    elif parameters['retailers'] == 9:
        return _integral9(rvs[0], rvs[1], rvs[2],
                          rvs[3], rvs[4], rvs[5],
                          rvs[6], rvs[7], rvs[8],
                          parameters['Q1'], parameters['Q2'], parameters['Q3'],
                          parameters['Q4'], parameters['Q5'], parameters['Q6'],
                          parameters['Q7'], parameters['Q8'], parameters['Q9'], target)
    else:
        print "Not implemented with {} retailers".format(parameters['retailers'])
        exit(1)

    # return _integral(rvs, [vars()['Q{}'.format(i)] for i in xrange(1, R+1)], R, target)


# @jit(nopython=True)
def _integral(rvs, Q, R, target):
    hit = 0.
    N = len(rvs[0])

    for t in xrange(N):
        if sum([MIN(rvs[i][t], Q[i]) for i in xrange(R)]) >= target:
            hit += 1.

    return hit/N


@jit(nopython=True)
def _integral3(rv1, rv2, rv3, Q1, Q2, Q3, target):
    hit = 0.
    N = len(rv1)
    for i in xrange(N):
        if MIN(rv1[i], Q1) + MIN(rv2[i], Q2) + MIN(rv3[i], Q3) >= target:
            hit += 1.

    return hit/N


@jit(nopython=True)
def _integral6(rv1, rv2, rv3, rv4, rv5, rv6, Q1, Q2, Q3, Q4, Q5, Q6, target):
    hit = 0.
    N = len(rv1)
    for i in xrange(N):
        if MIN(rv1[i], Q1) + MIN(rv2[i], Q2) + MIN(rv3[i], Q3) + MIN(rv4[i], Q4) + MIN(rv5[i], Q5) + MIN(rv6[i], Q6) >= target:
            hit += 1.

    return hit/N


@jit(nopython=True)
def _integral9(rv1, rv2, rv3, rv4, rv5, rv6, rv7, rv8, rv9, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, target):
    hit = 0.
    N = len(rv1)
    for i in xrange(N):
        if MIN(rv1[i], Q1) + MIN(rv2[i], Q2) + MIN(rv3[i], Q3) +\
           MIN(rv4[i], Q4) + MIN(rv5[i], Q5) + MIN(rv6[i], Q6) +\
           MIN(rv7[i], Q7) + MIN(rv8[i], Q8) + MIN(rv9[i], Q9) >= target:
            hit += 1.

    return hit/N


def f3TruncNormRVSnp(parameters):
    N = parameters['N']
    target = parameters['target']
    rv1, rv2, rv3 = ndarray(shape = (N,), dtype=float), ndarray(shape = (N,), dtype=float), ndarray(shape = (N,), dtype=float)

    # if parameters['ncpu']:
    #     ncpu = parameters['ncpu']
    # else:
    #     ncpu = mp.cpu_count()
    #
    # pool = mp.Pool(ncpu)
    # workers = []
    if not parameters['distribution']:
        print 'No distribution set...abort'
        exit(1)
    elif parameters['distribution'] == 'truncnorm':
        a1, b1 = (parameters['min_intrv1'] - parameters['mu1']) / parameters['sigma1'], (parameters['max_intrv1'] - parameters['mu1']) / parameters['sigma1']
        a2, b2 = (parameters['min_intrv2'] - parameters['mu2']) / parameters['sigma2'], (parameters['max_intrv2'] - parameters['mu2']) / parameters['sigma2']
        a3, b3 = (parameters['min_intrv3'] - parameters['mu3']) / parameters['sigma3'], (parameters['max_intrv3'] - parameters['mu3']) / parameters['sigma3']
        rv1 = truncnorm(a1, b1, loc=parameters['mu1'], scale=parameters['sigma1']).rvs(N)
        rv2 = truncnorm(a2, b2, loc=parameters['mu2'], scale=parameters['sigma2']).rvs(N)
        rv3 = truncnorm(a3, b3, loc=parameters['mu3'], scale=parameters['sigma3']).rvs(N)
    elif parameters['distribution'] == 'norm':
        rv1 = norm(loc=parameters['mu1'], scale=parameters['sigma1']).rvs(N)
        rv2 = norm(loc=parameters['mu2'], scale=parameters['sigma2']).rvs(N)
        rv3 = norm(loc=parameters['mu3'], scale=parameters['sigma3']).rvs(N)
    elif parameters['distribution'] == 'uniform':
        rv1 = uniform(loc=parameters['mu1'], scale=parameters['sigma1']).rvs(N)
        rv2 = uniform(loc=parameters['mu2'], scale=parameters['sigma2']).rvs(N)
        rv3 = uniform(loc=parameters['mu3'], scale=parameters['sigma3']).rvs(N)
    elif parameters['distribution'] == 'beta':
        rv1 = beta(a=parameters['min_intrv1'], b=parameters['max_intrv1'], loc=parameters['mu1'], scale=parameters['sigma1']).rvs(N)
        rv2 = beta(a=parameters['min_intrv2'], b=parameters['max_intrv2'], loc=parameters['mu2'], scale=parameters['sigma2']).rvs(N)
        rv3 = beta(a=parameters['min_intrv3'], b=parameters['max_intrv3'], loc=parameters['mu3'], scale=parameters['sigma3']).rvs(N)
    elif parameters['distribution'] == 'triang':
        rv1 = triang(loc=parameters['min_intrv1'], scale=parameters['max_intrv1'], c=parameters['mu1']).rvs(N)
        rv2 = triang(loc=parameters['min_intrv2'], scale=parameters['max_intrv2'], c=parameters['mu2']).rvs(N)
        rv3 = triang(loc=parameters['min_intrv3'], scale=parameters['max_intrv3'], c=parameters['mu3']).rvs(N)
    else:
        print 'Distribution not recognized...abort'
        exit(1)

    if parameters['scaling']:
        #scale the values of Qs in the allowed range such that sum(Q_i) = A
        r = ABS(parameters['Q1']) + ABS(parameters['Q2']) + ABS(parameters['Q3'])
        if r == 0.0:
            r = 1.

        # rounding the values, the sum could exceed A
        Q1 = ABS(parameters['Q1']) * parameters['A'] / r
        Q2 = ABS(parameters['Q2']) * parameters['A'] / r
        Q3 = parameters['A'] - Q1 - Q2
    else:
        # print "scaling = False"
        Q1 = parameters['Q1']
        Q2 = parameters['Q2']
        Q3 = parameters['Q3']

    return _f3(rv1, rv2, rv3, Q1, Q2, Q3, target)

    # Version with multiprocessing
    # _N = int(N/4.)
    # workers.append(pool.apply_async(_f3, args=(rv1[0:_N], rv2[0:_N], rv3[0:_N],Q1, Q2, Q3, target)))
    # workers.append(pool.apply_async(_f3, args=(rv1[_N:2*_N], rv2[_N:2*_N], rv3[_N:2*_N], Q1, Q2, Q3, target)))
    # workers.append(pool.apply_async(_f3, args=(rv1[2*_N:3*_N], rv2[2*_N:3*_N], rv3[2*_N:3*_N], Q1, Q2, Q3, target)))
    # workers.append(pool.apply_async(_f3, args=(rv1[3*_N:N], rv2[3*_N:N], rv3[3*_N:N], Q1, Q2, Q3, target)))
    #
    # hit = 0.0
    # for w in workers:
    #     hit += w.get()
    # pool.close()
    #
    # return float(hit/N)
    # call the compiled function that perform the calculation loop


@jit(nopython=True)
def _f3(rv1, rv2, rv3, Q1, Q2, Q3, target):
    hit = 0.
    N = len(rv1)
    # not_hit = 0

    for i in xrange(N):
        if MIN(rv1[i], Q1) + MIN(rv2[i], Q2) + MIN(rv3[i], Q3) >= target:
            hit += 1.
        # else:
        #     not_hit += 1.

    return hit/N

# def f3TruncNormRVSnp(parameters):
    # This version does not use the numba function _f3
#     N = parameters['N']
#     target = parameters['target']
#     rv1, rv2, rv3 = ndarray(shape = (N,), dtype=float), ndarray(shape = (N,), dtype=float), ndarray(shape = (N,), dtype=float)
#
#     a1, b1 = (parameters['min_intrv1'] - parameters['mu1']) / parameters['sigma1'], (parameters['max_intrv1'] - parameters['mu1']) / parameters['sigma1']
#     a2, b2 = (parameters['min_intrv2'] - parameters['mu2']) / parameters['sigma2'], (parameters['max_intrv2'] - parameters['mu2']) / parameters['sigma2']
#     a3, b3 = (parameters['min_intrv3'] - parameters['mu3']) / parameters['sigma3'], (parameters['max_intrv3'] - parameters['mu3']) / parameters['sigma3']
#     rv1 = truncnorm(a1, b1, loc=parameters['mu1'], scale=parameters['sigma1']).rvs(N)
#     rv2 = truncnorm(a2, b2, loc=parameters['mu2'], scale=parameters['sigma2']).rvs(N)
#     rv3 = truncnorm(a3, b3, loc=parameters['mu3'], scale=parameters['sigma3']).rvs(N)
#
#     if parameters['scaling']:
#         #scale the values of Qs in the allowed range such that sum(Q_i) = A
#         r = abs(parameters['Q1']) + abs(parameters['Q2']) + abs(parameters['Q3'])
#         if r == 0.0:
#             r = 1.
#
#         # rounding the values, the sum could exceed A
#         Q1 = trunc(ABS(parameters['Q1']) * parameters['A'] / r)
#         Q2 = trunc(ABS(parameters['Q2']) * parameters['A'] / r)
#         Q3 = parameters['A'] - Q1 - Q2
#     else:
#         # print "scaling = False"
#         Q1 = parameters['Q1']
#         Q2 = parameters['Q2']
#         Q3 = parameters['Q3']
#
#     hit = 0
#     not_hit = 0
#     print "NM Start"
#     for t in xrange(1, 4):
#         print "Q{}: {}".format(t, parameters['Q{}'.format(t)])
#
#     for i in range(N):
#         if MIN(rv1[i], Q1) + MIN(rv2[i], Q2) + MIN(rv3[i], Q3) >= target:
#             hit += 1.
#         else:
#             not_hit += 1.
#
#     print "NM End - P: {}".format(float(hit)/N)
#
#     return float(hit)/N


if __name__ == "__main__":

    for t in xrange(40, 131, 15):
        parameters['target'] = t
        p = f3TruncNormRVSnp(parameters)
        print "Target: {}  p: {}".format(t, p)




