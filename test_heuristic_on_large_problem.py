# Test the greedy heuristics with a problem with larger A

import pandas as pd
from truncnorm_custom import truncnorm_custom
from allocation import greedy_allocation3, parameters
from downhill_search import nm, f3wrapper
from random import random, randint
import time
import logging

logging.basicConfig(level=logging.DEBUG)

parameters['scaling'] = True
parameters['funcwrapper'] = f3wrapper
parameters['N'] = 200000
parameters['xtol'] = 1.
parameters['ftol'] = 0.1
parameters['numrun'] = 10
parameters['ncpu'] = 4

if __name__ == '__main__':

    df = pd.read_excel('Data_alloc_change_on_target_change_comp_NM_large_problem_test.xlsx')
    df_results = pd.read_excel('Results_alloc_change_on_target_change_comp_NM_large_problem.xlsx')

    df_results_columns = [u'DATASET', u'N', u'AVAIL', u'TARGET', u'MU1', u'S1', u'MU2', u'S2', u'MU3', u'S3', u'MAX_INTRV', u'NM_PROB', u'NM_TIME', u'NUM_ITER', u'FUN_EVAL', u'Q1', u'Q2', u'Q3', u'ALLOC_HEU', u'HEURISTIC_VALUE', u'HEURISTIC_TIME']

    for index, row in df.iterrows():
        # get from excel file
        logging.debug('Start index {} (dataset {})'.format(index, row['DATASET']))

        parameters['mu1'], parameters['mu2'], parameters['mu3'] = row['MU1'],row['MU2'],row['MU3']
        parameters['sigma1'], parameters['sigma2'], parameters['sigma3'] = row['S1'],row['S2'],row['S3']
        parameters['A'] = row['AVAIL']
        parameters['target'] = row['TARGET']
        parameters['max_intrv1'], parameters['max_intrv2'], parameters['max_intrv3'] = row['MAX_INTRV'], row['MAX_INTRV'], row['MAX_INTRV']
        parameters['dataset'] = row['DATASET']

        logging.debug('Start greedy heuristics')
        start_time = time.time()
        result_heu = greedy_allocation3(parameters)
        end_time = time.time()
        heu_time = end_time - start_time
        logging.debug('Greedy heuristics ended')

        numrun = parameters['numrun']
        for t in xrange(numrun):

            parameters['Q1'], parameters['Q2'], parameters['Q3'] = randint(100, 600), randint(100, 600), randint(100, 600)
            logging.debug('Start NM run {}'.format(t+1))
            start_time = time.time()
            result_nm = nm(parameters)
            end_time = time.time()
            nm_time = end_time - start_time
            logging.debug('NM run {} ended'.format(t+1))

            logging.debug('Create results DF')
            # df_results.ix[index, 'DATASET'] = row['DATASET']
            # df_results.ix[index, 'N'] = 3
            # df_results.ix[index, 'AVAIL'] = parameters['A']
            # df_results.ix[index, 'TARGET'] = parameters['target']
            # df_results.ix[index, 'MU1'] = parameters['mu1']
            # df_results.ix[index, 'MU2'] = parameters['mu2']
            # df_results.ix[index, 'MU3'] = parameters['mu3']
            # df_results.ix[index, 'S1'] = parameters['sigma1']
            # df_results.ix[index, 'S2'] = parameters['sigma2']
            # df_results.ix[index, 'S3'] = parameters['sigma3']
            # df_results.ix[index, 'MAX_INTRV'] = row['MAX_INTRV']
            # df_results.ix[index, 'NM_PROB'] = result_nm.fun
            # df_results.ix[index, 'NM_TIME'] = nm_time
            # df_results.ix[index, 'NUM_ITER'] = result_nm.nit
            # df_results.ix[index, 'FUN_EVAL'] = result_nm.nfev
            # df_results.ix[index, 'Q1'] = result_nm.x[0]
            # df_results.ix[index, 'Q2'] = result_nm.x[1]
            # df_results.ix[index, 'Q3'] = result_nm.x[2]
            # df_results.ix[index, 'ALLOC_HEU'] = str([result_heu['Q1'], result_heu['Q2'], result_heu['Q3']])
            # df_results.ix[index, 'HEURISTIC_VALUE'] = result_heu['PROB']
            # df_results.ix[index, 'HEURISTIC_TIME'] = heu_time

            tot = sum(result_nm.x)
            df_run_result = pd.DataFrame([[row['DATASET'],
                                          3,
                                          parameters['A'],
                                          parameters['target'],
                                          parameters['mu1'],
                                          parameters['sigma1'],
                                          parameters['mu2'],
                                          parameters['sigma2'],
                                          parameters['mu3'],
                                          parameters['sigma3'],
                                          row['MAX_INTRV'],
                                          result_nm.fun,
                                          nm_time,
                                          result_nm.nit,
                                          result_nm.nfev,
                                          result_nm.x[0]*parameters['A']/tot,
                                          result_nm.x[1]*parameters['A']/tot,
                                          result_nm.x[2]*parameters['A']/tot,
                                          str([result_heu['Q1'], result_heu['Q2'], result_heu['Q3']]),
                                          result_heu['PROB'],
                                          heu_time]], columns=df_results_columns)

            df_results=df_results.append(df_run_result, ignore_index=True)

    logging.debug('Write on disk')
    df_results.to_excel('Results.xlsx')

    logging.debug('Program complete')
