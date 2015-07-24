# -*- coding: utf-8 -*-

import multiprocessing as mp
import sys
import argparse
from MCIntegrals_numba import f3TruncNormRVSnp
from datetime import datetime
from downhill import downhill
from random import random

import decimal

context = decimal.getcontext()
context.prec = 3
context.rounding = getattr(decimal, 'ROUND_HALF_DOWN')

def fn(A, gridstep, t):
        """ stima il tempo necessario a completare il calcolo su un singolo processore che impiega t secondi per puntiìo
        A = limite superiore da testare
        gridstep = step sulla griglia
        t = secondi per calcolo dell'integrale
        """
        hit = 0
        for i in xrange(0, A, gridstep):
                for j in xrange(0, A - i + 1, gridstep):
                        hit += 1

        est_time = (((hit * t) / 60) / 60)
        print "Estimated time @ %f per point: %f hours (single processor)" % (t, est_time)


def mypool(parameters, lb, ub, wid):
        # print "Start lb: ", lb, " ub: ", ub
        # print "parameters received by mypool: ", id(parameters)
        print "Starting worker %d..." % wid
        results = {}
        best = {'worker': wid, 'allocation': (-1, -1, -1), 'value': 0.0}

        A = int(parameters['A'])
        # print "A: ", A
        # counter = 0


        for i in xrange(lb, ub, parameters['gridstep']):
                for j in xrange(0, A - i + 1, parameters['gridstep']):

                        k = A - i - j
                        # print "--->i, j, k: ", i,", ",j,", ", k, ", ", i+j+k
                        # counter += 1
                        if i + j + k != A:
                                print "ERROR: ", i + j + k, " LB: ", lb, " UB: ", ub

                        # from integer to float, faster than float(i)
                        parameters['Q1'] = 1. * i
                        parameters['Q2'] = 1. * j
                        parameters['Q3'] = 1. * k

                        results[(i, j, k)] = f3TruncNormRVSnp(parameters)
                        if best['value'] < results[(i, j, k)]:
                                # print "old value: ", best['value'], " vs new value: ",results[(i,j,k)]
                                best['value'] = results[(i, j, k)]
                                best['allocation'] = (i, j, k)
                                # print "update best value from worker %d: %f @ %d %d %d" % (wid, best['value'], i,j,k)

        print "Worker %d completed!" % wid
        return best


def run_pool(parameters):
        # print "parameters received by runpool: ", id(parameters)
        pool = mp.Pool(mp.cpu_count())
        workers = []
        worker_count = 1
        parameters['scaling'] = False

        if parameters['A'] % parameters['step'] != 0:
                print "A = %f not divisible by %d" % (
                        parameters['A'], parameters['step'])
                sys.exit(1)

        print "Start parallel processes..."
        A = int(parameters['A'])
        step = int(parameters['step'])
        for i in xrange(0, A, step):
                # print "i: ", i, "i + step: ",(i+step), " A: ", A," step: ", step
                # print "data sent from runpool: ", id(parameters)

                workers.append(pool.apply_async(mypool, args = (parameters, i, i + step, worker_count)))
                worker_count += 1

        # results = [r.get() for r in workers]

        # check the best value
        best = 0
        results = []
        for r in workers:
                rr = r.get()
                if rr['value'] > best:
                        results = rr
                        best = rr['value']
        pool.close()
        # best = max(results.iterkeys(), key=(lambda key: results[key]))
        return results

def f3wrapper(x0, parameters):
        parameters['Q1'] = x0[0]
        parameters['Q2'] = x0[1]
        parameters['Q3'] = x0[2]
        return -1. * f3TruncNormRVSnp(parameters)

# def f6wrapper(x0, parameters):
#         parameters['Q1'] = x0[0]
#         parameters['Q2'] = x0[1]
#         parameters['Q3'] = x0[2]
#         parameters['Q4'] = x0[3]
#         parameters['Q5'] = x0[4]
#         parameters['Q6'] = x0[5]
#
#         return -1. * f6TruncNormRVSnp(parameters)
#
#
# def f9wrapper(x0, parameters):
#         parameters['Q1'] = x0[0]
#         parameters['Q2'] = x0[1]
#         parameters['Q3'] = x0[2]
#         parameters['Q4'] = x0[3]
#         parameters['Q5'] = x0[4]
#         parameters['Q6'] = x0[5]
#         parameters['Q7'] = x0[6]
#         parameters['Q8'] = x0[7]
#         parameters['Q9'] = x0[8]
#
#         return -1. * f9TruncNormRVSnp(parameters)


def my_callback(x):
        print x

def nm(parameters):
        # first guess
        print "Starting downhill"

        results = []
        lQ =  ['Q{}'.format(i) for i in xrange(1, parameters['retailers']+1)]

        for k in xrange(parameters['repeat']):
                print "Run ", k+1
                parameters['scaling'] = True

                # TODO: improve the randomization of starting point
                if k == 0:
                        # x0 = [parameters['Q1'], parameters['Q2'], parameters['Q3']]
                        x0 = [parameters[retailer] for retailer in lQ]
                else:
                        # x0 = [random(), random(), random()]
                        x0 = [random() for retailer in lQ]

                results.append(downhill(parameters['funcwrapper'], x0, args=parameters,
                                       side = parameters['simplexsize'],
                                       xtol = parameters['xtol'],
                                       ftol = parameters['ftol'],
                                       maxiter = parameters['maxiter'],
                                       maxfunc = parameters['maxfunc'],
                                       maxiternochange = parameters['maxiternochange']
                                        )
                               )

        return results


if __name__ == '__main__':
        parser = argparse.ArgumentParser(
                description = 'Probability exhaustive search')
        parser.add_argument('-m', '--mu', dest = 'mu', required = True,
                            type = float, nargs = 3, default = [0., 0., 0.],
                            help = 'Medie della domanda')
        parser.add_argument('-s', '--sigma', dest = 'sigma', required = True,
                            type = float, nargs = 3, default = [0., 0., 0.],
                            help = 'Dev. std. della domanda')
        parser.add_argument('-l', '--minintrv', dest = 'minintrv',
                            required = True, type = float, nargs = 3,
                            default = [0., 0., 0.],
                            help = 'Valore minimo della domanda')
        parser.add_argument('-u', '--maxintrv', dest = 'maxintrv',
                            required = True, type = float, nargs = 3,
                            default = [0., 0., 0.],
                            help = 'Valore massimo della domanda')
        parser.add_argument('-t', '--target', dest = 'target', required = True,
                            type = float, default = 0., help = 'Target')
        parser.add_argument('-A', '--availability', dest = 'A',
                            required = True, type = float, default = 0.,
                            help = 'Disponibilità')
        parser.add_argument('-p', '--price', dest = 'price', type = float,
                            default = 1., help = 'Prezzo')
        parser.add_argument('-N', '--iterations', dest = 'N', type = int,
                            default = 200000, help = 'Numero iterazioni')
        parser.add_argument('-d', '--step', dest = 'step', required = True,
                            type = int, help = 'Step')
        parser.add_argument('-g', '--gridstep', dest = 'gridstep',
                            required = True, type = int, default = 10,
                            help = 'Gridstep')
        parser.add_argument('-i', '--initQ', dest = 'initQ', required = True,
                            type = float, nargs = 3, default = [1., 1., 1.],
                            help = 'Valore iniziale di Q')
        parser.add_argument('-f','--outfile', dest = 'outfile', type = str,
                            default = 'output.csv', help = 'Output file for the results')
        parser.add_argument('--xtol', dest = 'xtol', type = float,
                            default = 0.001, help = 'xtol')
        parser.add_argument('--ftol', dest = 'ftol', type = float,
                            default = 0.0001, help = 'ftol')
        parser.add_argument('--maxiter', dest = 'maxiter', type = int,
                            default = 2000, help = 'NM maxiter')
        parser.add_argument('--maxfunc', dest = 'maxfunc', type = int,
                            default = 2000, help = 'NM maxfunc')
        parser.add_argument('--simplexsize', dest = 'simplexsize', type = float,
                            default = 0.05, help = 'Simplex size')
        parser.add_argument('--maxiternochange', dest = 'maxiternochange', type = int,
                            default = 20, help = 'Maximum number of allowed iteration with no change in the best solution')
        parser.add_argument('-r','--repeat', dest = 'repeat', type = int,
                            default = 10, help = 'Number of trials for the downhill algorithm')
        parser.add_argument('-R', '--retailers', dest = 'retailers', type = int,
                            default = 3, help = 'Number of retailers (size of the problem')


        args = parser.parse_args()

        parameters = {'Q1': args.initQ[0],
                      'Q2': args.initQ[1],
                      'Q3': args.initQ[2],
                      'mu1': args.mu[0],
                      'mu2': args.mu[1],
                      'mu3': args.mu[2],
                      'sigma1': args.sigma[0],
                      'sigma2': args.sigma[1],
                      'sigma3': args.sigma[2],
                      'min_intrv1': args.minintrv[0],
                      'min_intrv2': args.minintrv[1],
                      'min_intrv3': args.minintrv[2],
                      'max_intrv1': args.maxintrv[0],
                      'max_intrv2': args.maxintrv[1],
                      'max_intrv3': args.maxintrv[2],
                      'price': args.price,
                      'target': args.target,
                      'A': args.A,
                      'N': args.N,
                      'step': args.step,
                      'gridstep': args.gridstep,  # finezza della griglia
                      'scaling': False,
                      'xtol': args.xtol,
                      'ftol': args.ftol,
                      'maxiter': args.maxiter,
                      'maxfunc': args.maxfunc,
                      'simplexsize': args.simplexsize,
                      'maxiternochange': args.maxiternochange,
                      'outfile': args.outfile,
                      'repeat': args.repeat,
                      'retailers': args.retailers,
                      'funcwrapper': f3wrapper
        }

        t_start = datetime.now()
        parameters['resultsdownhill'] = nm(parameters)
        t_end = datetime.now()
        delta = t_end - t_start
        parameters['timedownhill'] = delta.total_seconds()
        print "Elapsed time downhill: ", parameters['timedownhill']
        print parameters['resultsdownhill']
        exit(1)
        t_start = datetime.now()
        parameters['resultsexhaustive'] = run_pool(parameters)
        t_end = datetime.now()
        delta = t_end - t_start
        parameters['timeexhaustive'] = delta.total_seconds()
        print "Elapsed time exhaustive search: ", parameters['timeexhaustive']
        print parameters['resultsexhaustive']


        # TODO: print on file...
        print "print on ", parameters['outfile']
        # print parameters
        # parameters = {'Q1': 500.,
        # 'Q2': 500.,
        # 'Q3': 500.,
        #       'mu1': 500.,
        #       'mu2': 500.,
        #       'mu3': 500.,
        #       'sigma1': 50.,
        #       'sigma2': 50.,
        #       'sigma3': 50.,
        #       'min_intrv1': 0.,
        #       'min_intrv2': 0.,
        #       'min_intrv3': 0.,
        #       'max_intrv1': 1000.,
        #       'max_intrv2': 1000.,
        #       'max_intrv3': 1000.,
        #       'price': 1.,
        #       'target': 1425.,
        #       'A': 1500.,
        #       'N': 100000,
        #       'step': 150,
        #       'gridstep': 10    # finezza della griglia
        #       }
        #
