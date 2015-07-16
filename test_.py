from multiallocation import truncnorm
from scipy.stats import truncnorm as sp_truncnorm
from nose.tools import assert_almost_equals
from random import randint


def test_truncnorm():
    a = 10.
    b = 100.
    mu = 25.
    sigma = 10.

    min_trunc, max_trunc = (a - mu) / sigma, (b - mu) / sigma

    sp_rv = sp_truncnorm(min_trunc, max_trunc, loc=mu, scale=sigma)
    rv = truncnorm(a,b,mu,sigma)

    for _ in xrange(100):
        n = randint(a, b)
        assert_almost_equals(sp_rv.cdf(n), rv.cdf(n), places=10)
        assert_almost_equals(sp_rv.pdf(n), rv.pdf(n), places=10)



