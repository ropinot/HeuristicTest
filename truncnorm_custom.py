from scipy.stats import norm

class truncnorm_custom(object):
    """
    https://en.wikipedia.org/wiki/Truncated_normal_distribution
    http://www.ntrand.com/truncated-normal-distribution/
    """
    def __init__(self, a, b, mu, sigma):
        self.a = a
        self.b = b
        self.mu = mu
        self.sigma = sigma
        self.A = (a - mu) / sigma
        self.B = (b - mu) / sigma
        self.Z = norm.cdf(self.B) - norm.cdf(self.A)

    def pdf(self, x):
        csi = (x - self.mu) / self.sigma
        return norm.pdf(csi) / (self.sigma * self.Z)


    def cdf(self, x):
        csi = (x - self.mu) / self.sigma
        return (norm.cdf(csi) - norm.cdf(self.A)) / self.Z
