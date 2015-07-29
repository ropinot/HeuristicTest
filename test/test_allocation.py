from nose.tools import assert_equal
from ..allocation import greedy_allocation, parameters
from ..MCIntegrals_numba import integral, f3TruncNormRVSnp
from random import randint, random
from ..downhill_search import nm, integral_wrapper


def test_greedy_allocation():
    """
    Test the greedy_heuristic() with 3, 4 and 5 retailers
    """

    parameters['A'] = 600
    parameters['mu1'], parameters['mu2'], parameters['mu3'] = 250., 250., 250.
    parameters['sigma1'], parameters['sigma2'], parameters['sigma3'] = 25., 25., 25.
    parameters['distribution'] = 'norm'
    parameters['scaling'] = True

    r = greedy_allocation(parameters)
    assert_equal(r['Q1'], 200)
    assert_equal(r['Q2'], 200)
    assert_equal(r['Q3'], 200)

    parameters['A'] = 1200
    r = greedy_allocation(parameters)
    assert_equal(r['Q1'], 400)
    assert_equal(r['Q2'], 400)
    assert_equal(r['Q3'], 400)

    # change the mu of one retailer
    parameters['mu1'] = 500.
    r = greedy_allocation(parameters)
    assert_equal(r['Q1'], 567)
    assert_equal(r['Q2'], 317)
    assert_equal(r['Q3'], 316)

    # change the sigma of one retailer
    parameters['sigma1'] = 90.
    r = greedy_allocation(parameters)
    assert_equal(r['Q1'], 629)
    assert_equal(r['Q2'], 286)
    assert_equal(r['Q3'], 285)

    # add one retailer
    parameters['mu4'] = 300.
    parameters['sigma4'] = 150.
    parameters['retailers'] = 4
    r = greedy_allocation(parameters)
    assert_equal(r['Q1'], 469)
    assert_equal(r['Q2'], 241)
    assert_equal(r['Q3'], 241)
    assert_equal(r['Q4'], 249)

    parameters['A'] = 2200
    parameters['target'] = 1800
    parameters['mu1'], parameters['mu2'], parameters['mu3'], parameters['mu4'] = 300., 400., 500., 600.
    parameters['sigma1'], parameters['sigma2'], parameters['sigma3'], parameters['sigma4'] = 90., 180., 270., 360
    parameters['distribution'] = 'norm'
    r = greedy_allocation(parameters)
    assert_equal(r['Q1'], 340)
    assert_equal(r['Q2'], 480)
    assert_equal(r['Q3'], 620)
    assert_equal(r['Q4'], 760)

    parameters['A'] = 2200
    parameters['target'] = 1800
    parameters['mu5'] = 700.
    parameters['sigma5'] = 450.
    parameters['retailers'] = 5
    r = greedy_allocation(parameters)
    assert_equal(r['Q1'], 280)
    assert_equal(r['Q2'], 360)
    assert_equal(r['Q3'], 440)
    assert_equal(r['Q4'], 520)
    assert_equal(r['Q5'], 600)


def test_integral_3():
    """
    Test the integral() function with 3 retailers
    """
    epsilon = 0.005
    parameters['A'] = 1200
    parameters['target'] = 1000
    parameters['mu1'], parameters['mu2'], parameters['mu3'], parameters['mu4'] = 400., 300., 300., 300.
    parameters['sigma1'], parameters['sigma2'], parameters['sigma3'], parameters['sigma4'] = 190., 190., 190., 190.
    parameters['distribution'] = 'norm'
    parameters['retailers'] = 3
    parameters['scaling'] = True

    v1 = f3TruncNormRVSnp(parameters)
    v2 = integral(parameters)
    assert abs(v1 - v2) <= epsilon

    for i in xrange(10):
        parameters['mu1'], parameters['mu2'], parameters['mu3'] = [randint(250, 400)] * 3
        parameters['sigma1'], parameters['sigma2'], parameters['sigma3'] = [randint(50, 200)] * 3
        v1 = f3TruncNormRVSnp(parameters)
        v2 = integral(parameters)
        assert abs(v1 - v2) <= epsilon


def test_integral_6():
    """
    Test the integral() function with 6 retailers
    """
    epsilon = 0.005
    parameters['A'] = 2200
    parameters['target'] = 1500
    parameters['mu1'], parameters['mu2'], parameters['mu3'] = 400., 300., 300.
    parameters['mu4'], parameters['mu5'], parameters['mu6'] = 400., 300., 300.
    parameters['sigma1'], parameters['sigma2'], parameters['sigma3'] = 190., 190., 190.
    parameters['sigma4'], parameters['sigma5'], parameters['sigma6'] = 190., 190., 190.
    parameters['distribution'] = 'norm'
    parameters['retailers'] = 6
    parameters['scaling'] = False
    parameters['Q1'] = 300.
    parameters['Q2'] = 300.
    parameters['Q3'] = 300.
    parameters['Q4'] = 300.
    parameters['Q5'] = 300.
    parameters['Q6'] = 300.
    v = integral(parameters)
    assert abs(v - 0.43182) <= epsilon

    parameters['target'] = 1000
    v = integral(parameters)
    assert abs(v - 0.93976) <= epsilon

    parameters['target'] = 3000.
    parameters['mu1'], parameters['mu2'], parameters['mu3'] = 300., 400., 500.
    parameters['mu4'], parameters['mu5'], parameters['mu6'] = 600., 700., 800.
    parameters['Q1'] = 500.
    parameters['Q2'] = 600.
    parameters['Q3'] = 500.
    parameters['Q4'] = 500.
    parameters['Q5'] = 600.
    parameters['Q6'] = 700.
    v = integral(parameters)
    assert abs(v - 0.24906) <= epsilon

    parameters['sigma1'], parameters['sigma2'], parameters['sigma3'] = 90., 190., 290.
    parameters['sigma4'], parameters['sigma5'], parameters['sigma6'] = 290., 190., 90.
    parameters['Q1'] = 500.
    parameters['Q2'] = 600.
    parameters['Q3'] = 500.
    parameters['Q4'] = 500.
    parameters['Q5'] = 600.
    parameters['Q6'] = 700.
    v = integral(parameters)
    assert abs(v - 0.21608) <= epsilon


def test_integral_9():
    """
    Test the integral() function with 9 retailers
    """
    epsilon = 0.005
    parameters['target'] = 5000
    parameters['mu1'], parameters['mu2'], parameters['mu3'] = 300., 400., 500.
    parameters['mu4'], parameters['mu5'], parameters['mu6'] = 600., 700., 800.
    parameters['mu7'], parameters['mu8'], parameters['mu9'] = 600., 700., 800.
    parameters['sigma1'], parameters['sigma2'], parameters['sigma3'] = 90., 190., 290.
    parameters['sigma4'], parameters['sigma5'], parameters['sigma6'] = 290., 190., 90.
    parameters['sigma7'], parameters['sigma8'], parameters['sigma9'] = 290., 190., 90.
    parameters['distribution'] = 'norm'
    parameters['retailers'] = 9
    parameters['N'] = 200000
    parameters['scaling'] = False
    parameters['Q1'] = 500.
    parameters['Q2'] = 600.
    parameters['Q3'] = 500.
    parameters['Q4'] = 500.
    parameters['Q5'] = 600.
    parameters['Q6'] = 700.
    parameters['Q7'] = 500.
    parameters['Q8'] = 600.
    parameters['Q9'] = 700.

    v = integral(parameters)
    assert abs(v - 0.0211) <= epsilon

    parameters['target'] = 2000
    parameters['mu1'], parameters['mu2'], parameters['mu3'] = 300., 300., 300.
    parameters['mu4'], parameters['mu5'], parameters['mu6'] = 300., 300., 300.
    parameters['mu7'], parameters['mu8'], parameters['mu9'] = 300., 300., 300.
    parameters['Q1'] = 500.
    parameters['Q2'] = 600.
    parameters['Q3'] = 500.
    parameters['Q4'] = 500.
    parameters['Q5'] = 600.
    parameters['Q6'] = 700.
    parameters['Q7'] = 500.
    parameters['Q8'] = 600.
    parameters['Q9'] = 700.

    v = integral(parameters)
    assert abs(v - 0.851235) <= epsilon

    parameters['target'] = 4000
    parameters['Q1'] = 500.
    parameters['Q2'] = 600.
    parameters['Q3'] = 500.
    parameters['Q4'] = 500.
    parameters['Q5'] = 600.
    parameters['Q6'] = 700.
    parameters['Q7'] = 500.
    parameters['Q8'] = 600.
    parameters['Q9'] = 700.

    v = integral(parameters)
    assert abs(v - 0.00096) <= epsilon


def test_downhill_vs_greedy_3():
    """
    Test the difference between the downhill and the greedy on 3 retailers
    """
    epsilon = 0.005
    parameters['mu1'], parameters['mu2'], parameters['mu3'] = 350., 250., 400.
    parameters['sigma1'], parameters['sigma2'], parameters['sigma3'] = 120., 90., 200.
    parameters['A'] = 1500
    parameters['target'] = 1000
    parameters['distribution'] = 'norm'
    parameters['retailers'] = 3
    parameters['funcwrapper'] = integral_wrapper
    parameters['numrun'] = 1
    parameters['scaling'] = True

    greedy = greedy_allocation(parameters)
    downhill = nm(parameters)

    assert abs(greedy['PROB'] + downhill.fun) <= epsilon


def test_downhill_vs_greedy_6():
    """
    Test the difference between the downhill and the greedy on 6 retailers
    """
    epsilon = 0.005
    parameters['mu1'], parameters['mu2'], parameters['mu3'] = 350., 250., 400.
    parameters['mu4'], parameters['mu5'], parameters['mu6'] = 350., 250., 400.
    parameters['sigma1'], parameters['sigma2'], parameters['sigma3'] = 120., 90., 200.
    parameters['sigma4'], parameters['sigma5'], parameters['sigma6'] = 120., 90., 200.
    parameters['A'] = 3000
    parameters['target'] = 2000
    parameters['distribution'] = 'norm'
    parameters['retailers'] = 6
    parameters['funcwrapper'] = integral_wrapper
    parameters['numrun'] = 1
    parameters['scaling'] = True

    greedy = greedy_allocation(parameters)
    downhill = nm(parameters)

    assert abs(greedy['PROB'] + downhill.fun) <= epsilon # greedy vs downhill 6


def test_downhill():
    """
    Test downhill on same retailers
    """
    epsilon = 0.005
    parameters['mu1'], parameters['mu2'], parameters['mu3'] = 400., 400., 400.
    parameters['sigma1'], parameters['sigma2'], parameters['sigma3'] = 100., 100., 100.
    parameters['A'] = 1200
    parameters['target'] = 950
    parameters['distribution'] = 'norm'
    parameters['retailers'] = 3
    parameters['funcwrapper'] = integral_wrapper
    parameters['numrun'] = 10
    parameters['scaling'] = True
    parameters['xtol'] = 1.
    parameters['ftol'] = 0.01

    parameters['Q1'] = 100.
    parameters['Q2'] = 200.
    parameters['Q3'] = 300.

    downhill = nm(parameters)
    assert abs(0.888685 + downhill.fun) <= epsilon # T=950

    parameters['target'] = 1050
    downhill = nm(parameters)
    assert abs(0.665920 + downhill.fun) <= epsilon # T=1050

    best_p = 0.0
    for t in xrange(parameters['numrun']):
        parameters['Q1'], parameters['Q2'], parameters['Q3'] = random(), random(), random()
        parameters['target'] = 1050
        downhill = nm(parameters)
        if downhill.fun < best_p:
            best_p = downhill.fun

    assert abs(0.665920 + best_p) <= epsilon # T=1050 random

    # parameters['target'] = 1150
    # downhill = nm(parameters)
    # assert abs(0.299545 + downhill.fun) <= epsilon # T=1150

    best_p = 0.0
    for t in xrange(parameters['numrun']):
        parameters['Q1'], parameters['Q2'], parameters['Q3'] = random(), random(), random()
        parameters['target'] = 1150
        downhill = nm(parameters)
        if downhill.fun < best_p:
            best_p = downhill.fun

    assert abs(0.299545 + best_p) <= epsilon # T=1150 random
