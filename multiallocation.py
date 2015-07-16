from scipy.stats import norm
from MCIntegrals import *
import pandas as pd
import re

parameters = {'Q1': -1.0,
              'Q2': -1.0,
              'Q3': -1.0,
              'Q4': -1.0,
              'Q5': -1.0,
              'Q6': -1.0,
              'Q7': -1.0,
              'Q8': -1.0,
              'Q9': -1.0,
              'mu1': 0.0,
              'mu2': 0.0,
              'mu3': 0.0,
              'mu4': 0.0,
              'mu5': 0.0,
              'mu6': 0.0,
              'mu7': 0.0,
              'mu8': 0.0,
              'mu9': 0.0,
              'sigma1': 0.0,
              'sigma2': 0.0,
              'sigma3': 0.0,
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
              'max_intrv4': 100.,
              'max_intrv5': 100.,
              'max_intrv6': 100.,
              'max_intrv7': 100.,
              'max_intrv8': 100.,
              'max_intrv9': 100.,
              'target': 104.,
              'alpha': 0.5,
              'A': 130.,
              'N': 150000,
              'scaling': False,
              }


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



def split_data(data):
    """
    :param data: list of strings
    :return: a list of floats
    """

    tokens = re.findall(r"[\w']+", data)
    num_tokens = len(tokens)
    float_list = []
    for i in xrange(0, num_tokens, 2):
        float_list.append(float('{}.{}'.format(tokens[i], tokens[i+1])))

    return float_list


if __name__ == "__main__":

    df = pd.read_excel('Dati_random.xlsx')
    df['HEU_VALUE'] = 0.0
    df['HEU_ALLOC'] = 'X'

    for index, row in df.iterrows():
        print "Elaborating index {}".format(index)
        # read parameters from the excel file
        N = row['N']
        parameters['target'] = row['T']
        parameters['A'] = row['A']
        A = parameters['A']
        Q = {i:0 for i in xrange(N)}
        log = []

        # get the mu and sigma from the excel (get a list [mu1, sigma1, mu2, sigma2,...] )
        demand_params = split_data(row['DATA'])

        # assign the mu and sigma to the parameters
        j = 0
        for i in xrange(N):
            parameters['mu{}'.format(i+1)] = demand_params[j]
            parameters['sigma{}'.format(i+1)] = demand_params[j+1]
            j += 2


        # define the RVs
        rvs = []
        for i in xrange(N):
            # define the range of the random variables (only for truncated normal dist)
            # a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
            # a, b = (parameters['min_intrv{}'.format(i+1)] - parameters['mu{}'.format(i+1)]) / parameters['sigma{}'.format(i+1)], \
            #        (parameters['max_intrv{}'.format(i+1)] - parameters['mu{}'.format(i+1)]) / parameters['sigma{}'.format(i+1)]

            rvs.append(truncnorm_custom(a=parameters['min_intrv{}'.format(i+1)],
                                        b=parameters['max_intrv{}'.format(i+1)],
                                        mu=parameters['mu{}'.format(i+1)],
                                        sigma=parameters['sigma{}'.format(i+1)]))

        # Heuristic core
        while A > 0:
            best_probability = -1
            best_retailer = -1
            l = {} # log

            for n, r in enumerate(rvs):
                # calculate marginal probability
                p = 1 - r.cdf(Q[n]+1)
                if p < 0.0:
                    p = 0.0

                l[n] = p
                # print "n: {}   p: {}".format(n,p)

                # check for solution improvement
                if p > best_probability:
                    best_probability = p
                    best_retailer = n

            # print "Best retailer: {} with value {}".format(best_retailer, best_probability)
            # assign the marginal unit
            Q[best_retailer] += 1
            A -= 1
            log.append(l)

        # get the final solution
        for i in xrange(N):
            parameters['Q{}'.format(i+1)] = Q[i]

        print "Best allocation: {}".format(str(Q))
        # evaluate the real probability
        if N == 3:
            df.ix[index, 'HEU_VALUE'] = f3TruncNormRVSnp(parameters)
        elif N == 6:
            df.ix[index, 'HEU_VALUE'] = f6TruncNormRVSnp(parameters)
        elif N == 9:
            df.ix[index, 'HEU_VALUE'] = f9TruncNormRVSnp(parameters)

        df.ix[index, 'HEU_ALLOC'] = str([parameters['Q{}'.format(i+1)] for i in xrange(N)]).replace(',', ' /').replace('.', ',')
        print df
        exit(1)
        # if row['HEU_VALUE'] > row['MINP']:
        #     print "----- BETTER THAN TO DOWNHILL SOLUTION -----"
        # elif row['HEU_VALUE'] <= row['MINP']:
        #     print "+++++ WORSE THAN DOWNHILL SOLUTION +++++"
        # else:
        #     print "===== WORSE THAN DOWNHILL SOLUTION ====="
    print "Writing results on file..."
    df.to_excel('Dati_random_with_heuristics_results.xlsx')
    print "...done!"