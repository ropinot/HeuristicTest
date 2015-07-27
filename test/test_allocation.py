from nose.tools import assert_equal, assert_almost_equals
from ..allocation import greedy_allocation, parameters
from ..MCIntegrals_numba import integral, f3TruncNormRVSnp
from random import randint


def test_greedy_allocation():
    """
    Test the greedy_heuristic() with 3 and 4 retailers
    """

    parameters['A'] = 600
    parameters['mu1'], parameters['mu2'], parameters['mu3'] = 250., 250., 250.
    parameters['sigma1'], parameters['sigma2'], parameters['sigma3'] = 25., 25., 25.
    parameters['distribution'] = 'norm'

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


def test_integral():
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

    # assert_almost_equals(f3TruncNormRVSnp(parameters), integral(parameters), places=2)
    assert abs(f3TruncNormRVSnp(parameters) - integral(parameters)) <= epsilon

    for i in xrange(10):
        parameters['mu1'], parameters['mu2'], parameters['mu3'], parameters['mu4'] = [randint(250, 400)] * 4
        parameters['sigma1'], parameters['sigma2'], parameters['sigma3'], parameters['sigma4'] = [randint(50, 200)] * 4
        # assert_almost_equals(f3TruncNormRVSnp(parameters), integral(parameters), places=2)
        assert abs(f3TruncNormRVSnp(parameters) - integral(parameters)) <= epsilon