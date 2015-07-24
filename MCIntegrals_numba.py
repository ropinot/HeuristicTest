from scipy.stats import truncnorm
from numpy import ndarray, min
from numba import jit
from math import trunc


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


def f3TruncNormRVSnp(parameters):
    N = parameters['N']
    target = parameters['target']
    rv1, rv2, rv3 = ndarray(shape = (N,), dtype=float), ndarray(shape = (N,), dtype=float), ndarray(shape = (N,), dtype=float)

    a1, b1 = (parameters['min_intrv1'] - parameters['mu1']) / parameters['sigma1'], (parameters['max_intrv1'] - parameters['mu1']) / parameters['sigma1']
    a2, b2 = (parameters['min_intrv2'] - parameters['mu2']) / parameters['sigma2'], (parameters['max_intrv2'] - parameters['mu2']) / parameters['sigma2']
    a3, b3 = (parameters['min_intrv3'] - parameters['mu3']) / parameters['sigma3'], (parameters['max_intrv3'] - parameters['mu3']) / parameters['sigma3']
    rv1 = truncnorm(a1, b1, loc=parameters['mu1'], scale=parameters['sigma1']).rvs(N)
    rv2 = truncnorm(a2, b2, loc=parameters['mu2'], scale=parameters['sigma2']).rvs(N)
    rv3 = truncnorm(a3, b3, loc=parameters['mu3'], scale=parameters['sigma3']).rvs(N)

    if parameters['scaling']:
        #scale the values of Qs in the allowed range such that sum(Q_i) = A
        r = ABS(parameters['Q1']) + ABS(parameters['Q2']) + ABS(parameters['Q3'])
        if r == 0.0:
            r = 1.

        # rounding the values, the sum could exceed A
        Q1 = trunc(ABS(parameters['Q1']) * parameters['A'] / r)
        Q2 = trunc(ABS(parameters['Q2']) * parameters['A'] / r)
        Q3 = parameters['A'] - Q1 - Q2
    else:
        # print "scaling = False"
        Q1 = parameters['Q1']
        Q2 = parameters['Q2']
        Q3 = parameters['Q3']

    return _f3(rv1, rv2, rv3, Q1, Q2, Q3, target, N)


@jit(nopython=True)
def _f3(rv1, rv2, rv3, Q1, Q2, Q3, target, N):
    hit = 0
    not_hit = 0
    for i in range(N):
        if MIN(rv1[i], Q1) + MIN(rv2[i], Q2) + MIN(rv3[i], Q3) >= target:
            hit += 1.
        else:
            not_hit += 1.

    return float(hit)/N


# def f3TruncNormRVSnp(parameters):
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




