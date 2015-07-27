from nose.tools import assert_equal, assert_almost_equal
from ..allocation import greedy_allocation, parameters

def test_greedy_allocation():
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

