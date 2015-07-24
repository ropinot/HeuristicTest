# -*- coding: utf-8 -*-
from downhill_search import nm, f3wrapper
from allocation import parameters
from random import random, randint

parameters['scaling'] = True
parameters['funcwrapper'] = f3wrapper
parameters['max_intrv1'], parameters['max_intrv2'], parameters['max_intrv3'] = 1000.0, 1000.0, 1000.
parameters['mu1'], parameters['mu2'], parameters['mu3'] = 200., 500., 750.
parameters['sigma1'], parameters['sigma2'], parameters['sigma3'] = 125., 375., 187.50
parameters['A'] = 2000
parameters['N'] = 200000
parameters['target'] = 1500.
parameters['xtol'] = 1.
parameters['ftol'] = 1.

x = [random() for _ in xrange(3)]
parameters['Q1'], parameters['Q2'], parameters['Q3'] = randint(100, 1000), randint(100, 1000), randint(100, 1000)

for t in xrange(1, 4):
    print "Q{}: {}".format(t, parameters['Q{}'.format(t)])

# print f3wrapper([parameters['Q1'], parameters['Q2'], parameters['Q3']], parameters)

r = nm(parameters)

print r