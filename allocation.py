from scipy.stats import truncnorm
from MCIntegrals_numba import f3TruncNormRVSnp
import pandas as pd
from truncnorm_custom import truncnorm_custom

def heu_allocation3(df, parameters):
    """
    Greedy heuristic for 3 supplier
    :param df: dataframe containing the data from the excel file
    :param parameters: parameters dict
    :return: write o the df and save on the file
    """

    for index, row in df.iterrows():
        print "Elaborating index {}".format(index)
        # read parameters from the excel file
        parameters['mu1'] = row['MU1']
        parameters['mu2'] = row['MU2']
        parameters['mu3'] = row['MU3']
        parameters['sigma1'] = row['S1']
        parameters['sigma2'] = row['S2']
        parameters['sigma3'] = row['S3']
        parameters['target'] = row['TARGET']
        parameters['A'] = row['AVAIL']

        rv1 = truncnorm_custom(parameters['min_intrv1'], parameters['max_intrv1'], parameters['mu1'], parameters['sigma1'])
        rv2 = truncnorm_custom(parameters['min_intrv2'], parameters['max_intrv2'], parameters['mu2'], parameters['sigma2'])
        rv3 = truncnorm_custom(parameters['min_intrv3'], parameters['max_intrv3'], parameters['mu3'], parameters['sigma3'])

        A = parameters['A']
        i = 0
        Q = {i:0 for i in xrange(3)}
        log = []

        while A >= 0:
            best_probability = -1
            best_retailer = -1
            l = {}
            for n, r in enumerate([rv1, rv2, rv3]):
                p = 1 - r.cdf(Q[n]+1)
                l[n] = p
                # print "n: {}   p: {}".format(n,p)
                if p > best_probability:
                    best_probability = p
                    best_retailer = n

            # print "Best retailer: {} with value {}".format(best_retailer, best_probability)
            Q[best_retailer] += 1
            A -= 1
            # print "Current allocation: {}".format(Q)
            log.append(l)

        # print "best allocation:"
        # print Q
        parameters['Q1'] = Q[0]
        parameters['Q2'] = Q[1]
        parameters['Q3'] = Q[2]

        df.ix[index, 'ALLOC_HEU'] = str(Q.values())
        df.ix[index, 'HEURISTIC_VALUE'] = f3TruncNormRVSnp(parameters)


    df.to_excel(parameters['outfile'])


parameters = {'Q1': 1.,
              'Q2': 1.,
              'Q3': 1.,
              'mu1': 30.,
              'mu2': 70.,
              'mu3': 15.,
              'sigma1': 15.,
              'sigma2': 35.,
              'sigma3': 7.5,
              'min_intrv1': 0.,
              'min_intrv2': 0.,
              'min_intrv3': 0.,
              'max_intrv1': 100.,
              'max_intrv2': 100.,
              'max_intrv3': 100.,
              'price': 1.,
              'target': 104.,
              'alpha': 0.5,
              'A': 130.,
              'N': 150000,
              'step': 13,
              'gridstep': 1,  # finezza della griglia
              'scaling': False,
              'xtol': 0.001,
              'ftol': 0.00001,
              'maxiter': 300,
              'maxfunc': 900,
              'simplexsize': 0.05,
              'maxiternochange': 30,
              'retailers': 3,
              'outfile': '',
              'funcwrapper': None
              }



if __name__ == "__main__":
    df = pd.read_excel('ex_6_optimal_results.xlsx')

    for index, row in df.iterrows():
        print "Elaborating index {}".format(index)
        # read parameters from the excel file
        parameters['mu1'] = row['MU1']
        parameters['mu2'] = row['MU2']
        parameters['mu3'] = row['MU3']
        parameters['sigma1'] = row['S1']
        parameters['sigma2'] = row['S2']
        parameters['sigma3'] = row['S3']
        parameters['target'] = row['TARGET']
        parameters['A'] = row['AVAIL']

        # define the range of the random variables (only for truncated normal dist)
        # a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
        a1, b1 = (parameters['min_intrv1'] - parameters['mu1']) / parameters['sigma1'], (parameters['max_intrv1'] - parameters['mu1']) / parameters['sigma1']
        a2, b2 = (parameters['min_intrv2'] - parameters['mu2']) / parameters['sigma2'], (parameters['max_intrv2'] - parameters['mu2']) / parameters['sigma2']
        a3, b3 = (parameters['min_intrv3'] - parameters['mu3']) / parameters['sigma3'], (parameters['max_intrv3'] - parameters['mu3']) / parameters['sigma3']

        # define the RVs
        rv1 = truncnorm(a1, b1, loc=parameters['mu1'], scale=parameters['sigma1'])
        rv2 = truncnorm(a2, b2, loc=parameters['mu2'], scale=parameters['sigma2'])
        rv3 = truncnorm(a3, b3, loc=parameters['mu3'], scale=parameters['sigma3'])

        A = parameters['A']
        i = 0
        Q = {i:0 for i in xrange(3)}
        log = []

        while A >= 0:
            best_probability = -1
            best_retailer = -1
            l = {}
            for n, r in enumerate([rv1, rv2, rv3]):
                p = 1 - r.cdf(Q[n]+1)
                l[n] = p
                # print "n: {}   p: {}".format(n,p)
                if p > best_probability:
                    best_probability = p
                    best_retailer = n

            # print "Best retailer: {} with value {}".format(best_retailer, best_probability)
            Q[best_retailer] += 1
            A -= 1
            # print "Current allocation: {}".format(Q)
            log.append(l)

        # print "best allocation:"
        # print Q
        parameters['Q1'] = Q[0]
        parameters['Q2'] = Q[1]
        parameters['Q3'] = Q[2]

        df.ix[index, 'ALLOC_HEU'] = str(Q.values())
        df.ix[index, 'HEURISTIC_VALUE'] = f3TruncNormRVSnp(parameters)


    df.to_excel('ex_6_with_heuristics_results.xlsx')
