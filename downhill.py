#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import sqrt
from functools import partial
from numba import jit

import sys


# TODO: generalizzare la soluzione ritornata (allocation) su un numero parametrico di retailer
#@jit(nopython=True)
def downhill(F, xStart, args=None, side=0.1, ftol=1.0e-6, xtol=1.0e-6, maxiter=1000, maxfunc=1000, maxiternochange=10):
        """
        Downhill algorithm as described in Kiusalaas
        Numerical Methods in Engineering with Python 3

        :param F: function to be optimized, F(args)
        :param xtol: tolerance
        :param ftol: tolerance on the difference between best and worst value
        :param xStart: initial guess, np.array or list
        :param args: further input for F
        :param side: size of the simplex
        :param maxiter: maximum number of iterations
        :param maxfunc: maximum number of function call
        :return: optimal point
        """
        # TODO: check the types of the input ???

        # print "Entering downhill"
        n = len(xStart)
        x = np.zeros((n+1, n), dtype=float)  #point null matrix, n+1 rows, n columns
        f = np.zeros(n+1, dtype=float)       # null vector, n+1 columns
        p_count = 0  # counter for detecting a plateau
        f_count = 0     # counter for the number of function call
        f_best_count = 0    # counter for the number of iterations in which the best solution does not change
        f_best_prev = 0.0   # holds the best value from the previous iteration
        epsilon = 0.001 # tolerance for considering two values as equal
        # max_iter_no_change = 10 # maximum number of accepted iterations with no change in the optimal solution
        precision = 2
        round_map = partial(round, ndigits=precision)  # partial function for rounding purposes

        # initial simplex
        x[0] = xStart
        for i in xrange(1, n+1):
                x[i] = xStart
                x[i,i-1] = xStart[i-1] + side

        # print "Evaluate the starting points"
        # compute the value of F at the vertices of the simplex
        for i in xrange(n+1):
                f[i] = F(x[i], args)
                # p_count += 1

        # main loop
        # print "Start iterating"
        for k in xrange(maxiter):

                # check the number of function calls
                if f_count > maxfunc:
                        print "Stopping criteria: maximum number of function calls"
                        print "Best solution so far: ", x[iLo], " value: ", f[iLo], " at iteration:", k
                        # return {'point' : x[iLo], 'value': f[iLo], 'iteration': k, 'funcalls': f_count, 'allocation': (args['Q1'], args['Q2'], args['Q3']), 'stopping': 'MAXFUNCALL'}
                        return {'point' : x[iLo], 'value': f[iLo], 'iteration': k, 'funcalls': f_count, 'allocation': [args['Q{}'.format(h)] for h in xrange(1, args['retailers']+1)], 'stopping': 'MAXFUNCALL'}

                # find the best and worst vertex (consider a minimization problem)
                iLo = np.argmin(f)      # best vertex
                iHi = np.argmax(f)      # worst vertex

                # print k," ", f[iLo]
                #
                # if f[iLo] < -0.310000:
                #         print f[iLo]
                #         print x[iLo]
                #         print x
                #         sys.exit(1)
                # print "k: ", k, " f_best_prev: ", f_best_prev, " f[iLo]: ", f[iLo], " f_best_count: ", f_best_count
                # print "Beginning of iteration: %4d  |  Best x: %4f %4f %4f  |  Best value: %f" % (k, x[iLo][0], x[iLo][1], x[iLo][2], f[iLo])
                # print "x: ", x, " f: ", f
                # print "========================================================================================="
                # check if the solution has changed from the previous iterations
                if f[iLo] < f_best_prev:
                        f_best_prev = f[iLo]
                        f_best_count = 0
                else:
                        f_best_count += 1

                if f_best_count > maxiternochange:
                        print "Stopping criteria: maximum number of iterations with no improvement in the best solution"
                        print "Best solution so far: ", x[iLo], " value: ", f[iLo], " at iteration:", k
                        # return {'point' : x[iLo], 'value': f[iLo], 'iteration': k, 'funcalls': f_count, 'allocation': (args['Q1'], args['Q2'], args['Q3']), 'stopping': 'NOIMPROVEMENT'}
                        return {'point' : x[iLo], 'value': f[iLo], 'iteration': k, 'funcalls': f_count, 'allocation': [args['Q{}'.format(h)] for h in xrange(1, args['retailers']+1)], 'stopping': 'NOIMPROVEMENT'}

                if abs(f[iLo] - f[iHi]) < ftol: # If difference between highest and lowest is smaller than ftol, return
                        print "Stopping criteria: difference between highest and lowest points is smaller than tolerance"
                        print "Best solution so far: ", x[iLo], " value: ", f[iLo], " at iteration:", k
                        # return {'point' : x[iLo], 'value': f[iLo], 'iteration': k, 'funcalls': f_count, 'allocation': (args['Q1'], args['Q2'], args['Q3']), 'stopping': 'MAXTOLERANCE'}
                        return {'point' : x[iLo], 'value': f[iLo], 'iteration': k, 'funcalls': f_count, 'allocation': [args['Q{}'.format(h)] for h in xrange(1, args['retailers']+1)], 'stopping': 'MAXTOLERANCE'}
                # compute the move vector d
                d = (-(n+1) * x[iHi] + np.sum(x, axis=0)) / n
                # print "d: ", d

                # check for convergence
                if sqrt(np.dot(d, d)/n) < xtol:          # length of the vector d
                        print "Stopping criteria: length of step d smaller than tolerance"
                        print "Best solution so far: ", x[iLo], " value: ", f[iLo], " at iteration:", k
                        # return {'point' : x[iLo], 'value': f[iLo], 'iteration': k, 'funcalls': f_count, 'allocation': (args['Q1'], args['Q2'], args['Q3']), 'stopping': 'SMALLSTEP'}
                        return {'point' : x[iLo], 'value': f[iLo], 'iteration': k, 'funcalls': f_count, 'allocation': [args['Q{}'.format(h)] for h in xrange(1, args['retailers']+1)], 'stopping': 'SMALLSTEP'}
                # try reflection
                xNew = np.array(map(round_map, x[iHi] + 2 * d))
                fNew = F(xNew, args)
                f_count += 1
                # print "Reflected point: ", xNew, " value: ", fNew

                # check for no improvement over the worst point
                # and for plateau condition
                if f[iHi] - epsilon <= fNew <= f[iHi] + epsilon:
                        p_count += 1
                        # print "No improvement here"

                        if p_count == n+2:      # we reflected all vertices with no improvement
                                print "Stopping criteria: Probably we landed on a plateau... exiting"  # TODO: restart instead of exiting
                                print "Best solution so far: ", x[iLo], " value: ", f[iLo], " at iteration:", k
                                # return {'point' : x[iLo], 'value': f[iLo], 'iteration': k, 'funcalls': f_count, 'allocation': (args['Q1'], args['Q2'], args['Q3']), 'stopping': 'PLATEAU'}
                                return {'point' : x[iLo], 'value': f[iLo], 'iteration': k, 'funcalls': f_count, 'allocation': [args['Q{}'.format(h)] for h in xrange(1, args['retailers']+1)], 'stopping': 'PLATEAU'}


                else:
                        p_count = 0


                if fNew <= f[iLo]:      # if the new value is better than the best so far,
                        x[iHi] = xNew   # substitute the worst vertex with the new one
                        f[iHi] = fNew

                        # try to expand the reflection
                        xNew = np.array(map(round_map, x[iHi] + d))
                        fNew = F(xNew, args)
                        f_count += 1
                        # print "Expanded point: ", xNew, " value: ", fNew

                        if fNew <= f[iHi]:              # in the original source version it is f[iLo] (?)
                                x[iHi] = xNew
                                f[iHi] = fNew
                else:
                        # try reflection again
                        if fNew <= f[iHi]:
                                x[iHi] = xNew
                                f[iHi] = fNew
                        else:
                                # try contraction
                                xNew = np.array(map(round_map, x[iHi] + 0.5 * d))
                                fNew = F(xNew, args)
                                f_count += 1
                                # print "Contracted point: ", xNew, " value: ", fNew

                                if fNew <= f[iHi]: # accept contraction
                                        x[iHi] = xNew
                                        f[iHi] = fNew
                                else:
                                        # shrink
                                        for i in xrange(len(x)):
                                                if i != iLo:
                                                        x[i] = np.array(map(round_map, x[i] - x[iLo] * 0.5))
                                                        f[i] = F(x[i], args)
                                                        f_count += 1

                # print "End of iteration: %4d  |  Best x: %4f %4f %4f  |  Best value: %f" % (k, x[iLo][0], x[iLo][1], x[iLo][2], f[iLo])
                # print "x: ", x, " f: ", f
                # print "*"*50
                # print ""



        print "Stopping criteria: maximum number of iterations"
        print "Best solution so far: ", x[iLo], " value: ", f[iLo], " at iteration:", k
        # return {'point' : x[iLo], 'value': f[iLo], 'iteration': k, 'funcalls': f_count, 'allocation': (args['Q1'], args['Q2'], args['Q3']), 'stopping': 'MAXITERATION'}
        return {'point' : x[iLo], 'value': f[iLo], 'iteration': k, 'funcalls': f_count, 'allocation': [args['Q{}'.format(h)] for h in xrange(1, args['retailers']+1)], 'stopping': 'MAXITERATION'}



# def return_solution(stopping=-1):
#         return


# def amoeba2(F, xStart, args=None, side=0.1, ftol=1.0e-6, xtol=1.0e-6, maxiter=1000):
#         """ Version translated from the c version nm.c
#
#         :param F: function to be optimized
#         :param xStart: initial guess
#         :param side: size of the simplex
#         :param xtol: tolerance
#         :param ftol: tolerance on the difference between best and worst value
#         :param maxiterations: maximum number of iterations
#         :return: optimal point
#         """
#
#         n = len(xStart)
#         x = np.zeros((n+1, n), dtype=float)  # null matrix, n+1 rows, n columns
#         f = np.zeros(n+1, dtype=float)       # null vector, n+1 columns
#

#         # assert(type(xStart), list)
#
#         # initial simplex
#         x[0] = xStart
#         for i in xrange(1, n+1):
#                 x[i] = xStart
#                 x[i,i-1] = xStart[i-1] + side
#
#         # compute the value of F at the vertices of the simplex
#         for i in xrange(n+1):
#                 f[i] = F(x[i], args)
#
#
#         # constant parameters
#         # alpha, beta, gamma, delta = 1., 2., 0.5, 0.5
#
#         # main loop
#         for k in xrange(maxiter):
#                 # find the best and worst vertex (consider a minimization problem)
#                 iLo = np.argmin(f)      # best vertex
#                 iHi = np.argmax(f)      # worst vertex
#                 # print "iLo: ", iLo," ",f[iLo], " iHi: ", iHi, " ", f[iHi]
#
#                 # check the second best vertex
#                 fw = f[iHi]     # temp value
#                 f[iHi] = f[iLo] # make the worst value equal to the best
#                 sHi = np.argmax(f)      # second-worst vertex
#                 f[iHi] = fw     # restore original situation
#
#                 # print "iHi: ", iHi, " sHi: ", sHi, "iLo: ", iLo
#                 # print "F(iHi): ", f[iHi], " F(sHi): ", f[sHi], "F(iLo): ", f[iLo]
#                 # sys.exit(1)
#                 print "Beginning of iteration: %4d  |  Best x: %4f %4f %4f  |  Best value: %f" % (k, x[iLo][0], x[iLo][1], x[iLo][2], f[iLo])
#                 print "x: ", x, " f: ", f
#                 print "========================================================================================="
#
#
#                 if abs(f[iLo] - f[iHi]) < ftol: # If difference between highest and lowest is smaller than ftol, return
#                         print "Stopping criteria: difference between highest and lowest points is smaller than tolerance"
#                         print "Best solution so far: ", x[iLo], " value: ", f[iLo], " at iteration:", k
#                         return x[iLo]
#
#                 # compute the move vector d
#                 d = (-(n+1) * x[iHi] + np.sum(x, axis=0)) / n
#                 # print "d: ", d
#
#                 # check for convergence
#                 # print "convergence: ", sqrt(np.dot(d, d)/n)
#
#                 if sqrt(np.dot(d, d)/n) < xtol:          # length of the vector d
#                         print "Stopping criteria: length of step d smaller than tolerance"
#                         print "Best solution: ", x[iLo], " value: ", f[iLo], " at iteration:", k
#                         return x[iLo]
#
#                 # calculate reflection of the worst point
#                 x_r = x[iHi] + 2 * d
#                 f_x_r = F(x_r, args)
#                 # print "f_x_r: ", f_x_r, " f[iLo]: ", f[iLo]
#
#                 if f[iLo] <= f_x_r < f[sHi]:
#                         x[iHi] = x_r
#                         f[iHi] = f_x_r
#                         # print "f[iLo] <= fNew < f[iHi]"
#
#                 if f_x_r < f[iLo]:
#                         # try to expand the reflection
#                         x_e = x[iHi] + d
#                         f_x_e = F(x_e, args)
#                         # print "fNew: ", fNew, " f[iLo]: ", f[iLo]
#                         # print "fNew < f[iLo]"
#
#                         if f_x_e < f_x_r:
#                                 x[iHi] = x_e
#                                 f[iHi] = f_x_e
#                         else:
#                                 x[iHi] = x_r
#                                 f[iHi] = f_x_r
#
#                 if f_x_r >= f[sHi]:
#                         if f_x_r < f[iHi] and f_x_r >= f[sHi]:
#                                 x_c = x[iHi] + 0.5 * d         # test the outside contraction
#                                 f_x_c = F(x_c, args)
#                                 # print "f[sHi] <= fNew < f[iHi]"
#                         else:
#                                 x_c = x[iHi] - 0.5 * d         # test the inside contraction
#                                 f_x_c = F(x_c, args)
#                                 # print "f[sHi] <= fNew < f[iHi]"
#
#                         if f_x_c < f[iHi]:
#                                 x[iHi] = x_c
#                                 f[iHi] = f_x_c
#
#                         else:   # shrink
#                                 for i in xrange(len(x)):
#                                         if i != iLo:
#                                                 x[i] = (x[i] - x[iLo]) * 0.5
#                                                 f[i] = F(x[i], args)
#
#
#                 print "End of iteration: %4d  |  Best x: %4f %4f %4f  |  Best value: %f" % (k, x[iLo][0], x[iLo][1], x[iLo][2], f[iLo])
#                 print "x: ", x, " f: ", f
#                 print "*"*50
#                 print ""
#
#
#         print("Max number of iteration reached")
#         print "Best solution: ", x[iLo], " value: ", f[iLo]
#         return x[iLo]


# def amoeba(F, xStart, args=None, side=0.1, ftol=1.0e-6, xtol=1.0e-6, maxiter=1000):
#         """ Vanilla version of the amoeba algorithm as described in Gao & Han
#         Implementing the Nelder-Mead simplex algorithm with adaptive parameters
#         Comput Optim Appl
#         DOI 10.1007/s10589-010-9329-3
#
#         :param F: function to be optimized
#         :param xStart: initial guess
#         :param side: size of the simplex
#         :param xtol: tolerance
#         :param ftol: tolerance on the difference between best and worst value
#         :param maxiterations: maximum number of iterations
#         :return: optimal point
#         """
#
#         n = len(xStart)
#         x = np.zeros((n+1, n), dtype=float)  # null matrix, n+1 rows, n columns
#         f = np.zeros(n+1, dtype=float)       # null vector, n+1 columns
#
#
#         # assert(type(xStart), list)
#
#         # initial simplex
#         x[0] = xStart
#         for i in xrange(1, n+1):
#                 x[i] = xStart
#                 x[i,i-1] = xStart[i-1] + side
#
#         # compute the value of F at the vertices of the simplex
#         for i in xrange(n+1):
#                 f[i] = F(x[i], args)
#
#         # constant parameters
#         # alpha, beta, gamma, delta = 1., 2., 0.5, 0.5
#
#         # main loop
#         for k in xrange(maxiter):
#                 # find the best and worst vertex (consider a minimization problem)
#                 iLo = np.argmin(f)      # best vertex
#                 iHi = np.argmax(f)      # worst vertex
#                 # print "iLo: ", iLo," ",f[iLo], " iHi: ", iHi, " ", f[iHi]
#
#                 # check the second best vertex
#                 fw = f[iHi]     # temp value
#                 f[iHi] = f[iLo] # make the worst value equal to the best
#                 sHi = np.argmax(f)      # second-worst vertex
#                 f[iHi] = fw     # restore original situation
#
#                 # print "iHi: ", iHi, " sHi: ", sHi
#
#
#                 if abs(f[iLo] - f[iHi]) < ftol: # If difference between highest and lowest is smaller than ftol, return
#                         print "Stopping criteria: difference between highest and lowest points is smaller than tolerance"
#                         print "Best solution so far: ", x[iLo], " value: ", f[iLo], " at iteration:", k
#                         return x[iLo]
#
#                 # compute the move vector d
#                 d = (-(n+1) * x[iHi] + np.sum(x, axis=0)) / n
#                 # print "d: ", d
#
#                 # check for convergence
#                 # print "convergence: ", sqrt(np.dot(d, d)/n)
#
#                 if sqrt(np.dot(d, d)/n) < xtol:          # length of the vector d
#                         print "Stopping criteria: length of step d smaller than tolerance"
#                         print "Best solution: ", x[iLo], " value: ", f[iLo], " at iteration:", k
#                         return x[iLo]
#
#                 # calculate reflection of the worst point
#                 x_r = x[iHi] + 2 * d
#                 f_x_r = F(x_r, args)
#                 # print "f_x_r: ", f_x_r, " f[iLo]: ", f[iLo]
#
#                 if f[iLo] <= f_x_r < f[iHi]:      # if the new value is better than the worst one,
#                         x[iHi] = x_r              # then substitute the worst vertex with the new one
#                         f[iHi] = f_x_r
#                         # print "f[iLo] <= fNew < f[iHi]"
#
#                 elif f_x_r < f[iLo]:
#                         # try to expand the reflection
#                         x_e = x[iHi] + d
#                         f_x_e = F(x_e, args)
#                         # print "fNew: ", fNew, " f[iLo]: ", f[iLo]
#                         # print "fNew < f[iLo]"
#
#                         if f_x_e < f_x_r:
#                                 x[iHi] = x_e
#                                 f[iHi] = f_x_e
#                         else:
#                                 x[iHi] = x_r
#                                 f[iHi] = f_x_r
#
#                 elif f[sHi] <= f_x_r < f[iHi]:
#                         x_o_c = x[iHi] + 0.5 * d         # test the outside contraction
#                         f_o_c = F(x_o_c, args)
#                         # print "f[sHi] <= fNew < f[iHi]"
#
#                         if f_o_c <= f_x_r:
#                                 x[iHi] = x_o_c
#                                 f[iHi] = f_o_c
#                         else:   # shrink
#                                 for i in xrange(len(x)):
#                                         if i != iLo:
#                                                 x[i] = (x[i] - x[iLo]) * 0.5
#                                                 f[i] = F(x[i], args)
#
#                 elif f_x_r >= f[iHi]:
#                         x_i_c = x[iHi] - 0.5 * d         # test the inside contraction
#                         f_i_c = F(x_i_c, args)
#                         # print "fNew >= f[iHi]"
#
#                         if f_i_c < f[iHi]:
#                                 x[iHi] = x_i_c
#                                 f[iHi] = f_i_c
#
#                         else:
#                                 # shrink
#                                 for i in xrange(len(x)):
#                                         if i != iLo:
#                                                 x[i] = (x[i] - x[iLo]) * 0.5
#                                                 f[i] = F(x[i], args)
#
#         print("Max number of iteration reached")
#         print "Best solution: ", x[iLo], " value: ", f[iLo]
#         return x[iLo]
