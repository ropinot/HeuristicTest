# -*- coding: utf-8 -*-
from scipy.optimize import minimize
from MCIntegrals_numba import f3TruncNormRVSnp


def f3wrapper(x0, parameters):
    parameters['Q1'] = x0[0]
    parameters['Q2'] = x0[1]
    parameters['Q3'] = x0[2]
    return -1. * f3TruncNormRVSnp(parameters)


def nm(parameters):
    """
    Nelder-Mean search using scipy minimize
    :param parameters:
    :return:
    """
    x0 = [parameters['Q{}'.format(i)] for i in xrange(1, 4)]
    print "Start search with:"
    for t in xrange(1, 4):
        print "Q{}: {}".format(t, x0[t-1])

    res = minimize(parameters['funcwrapper'], x0, args=parameters, method='Nelder-Mead',
                   options={'xtol': parameters['xtol'],
                            'ftol': parameters['ftol'],
                            'disp': True})

    return res

