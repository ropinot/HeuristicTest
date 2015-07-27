# Test the greedy heuristics with a problem with larger A

import pandas as pd
from truncnorm_custom import truncnorm_custom
from allocation import greedy_allocation3, parameters
from downhill_search import nm, f3wrapper
from downhill import downhill
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
    import argparse
    parser = argparse.ArgumentParser(description='Tester')
    parser.add_argument('-i', action='store', dest='inputfile', type=str, required=True)
    parser.add_argument('-o', action='store', dest='outputfile', type=str, required=True)
    parser.add_argument('-n', action='store', dest='numrun', type=int, default=10)

    args=parser.parse_args()
    inputfile = args.inputfile
    outputfile = args.outputfile
    parameters['numrun'] = args.numrun

    # df = pd.read_excel('Data_alloc_change_on_target_change_comp_NM_large_problem_test.xlsx')
    # df_results = pd.read_excel('Results_alloc_change_on_target_change_comp_NM_large_problem.xlsx')

    df = pd.read_excel(inputfile)
    df_results_columns = [u'DATASET', u'N', u'DIST', u'AVAIL', u'TARGET', u'MU1', u'S1', u'MU2', u'S2', u'MU3', u'S3', u'MIN_INTRV', u'MAX_INTRV', u'NM_PROB', u'NM_TIME', u'NUM_ITER', u'FUN_EVAL', u'Q1', u'Q2', u'Q3', u'ALLOC_HEU', u'HEURISTIC_VALUE', u'HEURISTIC_TIME', u'DOWNHILL_ALLOC', u'DOWNHILL_PROB', u'DOWNHILL_TIME', u'DOWNHILL_ITER', u'DOWNHILL_FUNC']
    df_results = pd.DataFrame(columns=df_results_columns)

    for index, row in df.iterrows():
        # get from excel file
        logging.debug('Start index {} (dataset {})'.format(index, row['DATASET']))
        parameters['distribution'] = row['DIST']
        parameters['mu1'], parameters['mu2'], parameters['mu3'] = row['MU1'],row['MU2'],row['MU3']
        parameters['sigma1'], parameters['sigma2'], parameters['sigma3'] = row['S1'],row['S2'],row['S3']
        parameters['A'] = row['AVAIL']
        parameters['target'] = row['TARGET']
        parameters['min_intrv1'], parameters['min_intrv2'], parameters['min_intrv3'] = row['MIN_INTRV'], row['MIN_INTRV'], row['MIN_INTRV']
        parameters['max_intrv1'], parameters['max_intrv2'], parameters['max_intrv3'] = row['MAX_INTRV'], row['MAX_INTRV'], row['MAX_INTRV']
        parameters['dataset'] = row['DATASET']

        logging.debug('Start greedy heuristics')
        logging.debug('Distribution: {}'.format(parameters['distribution']))
        start_time = time.time()
        result_heu = greedy_allocation3(parameters)
        end_time = time.time()
        heu_time = end_time - start_time
        logging.debug('Greedy heuristics ended with P: {} and allocation {}'.format(result_heu['PROB'],
                                                                                    str([result_heu['Q1'],
                                                                                         result_heu['Q2'],
                                                                                         result_heu['Q3']])))

        numrun = parameters['numrun']
        for t in xrange(numrun):

            parameters['Q1'], parameters['Q2'], parameters['Q3'] = randint(100, 600), randint(100, 600), randint(100, 600)
            logging.debug('Start NM run {}/{}'.format(t+1, parameters['numrun']))
            start_time = time.time()
            result_nm = nm(parameters)
            end_time = time.time()
            nm_time = end_time - start_time
            logging.debug('NM run {} ended'.format(t+1))

            logging.debug('Downhill start run {}/{}'.format(t+1, parameters['numrun']))
            start_time = time.time()
            result_downhill = downhill(parameters['funcwrapper'],
                                       xStart=[parameters['Q1'], parameters['Q2'], parameters['Q3']],
                                       args=parameters)
            end_time = time.time()
            downhill_time = end_time - start_time

            logging.debug('Downhill run {} end with P: {}'.format(t+1, result_downhill['value']))

            logging.debug('Create results DF')

            tot = sum(result_nm.x)
            df_run_result = pd.DataFrame([[row['DATASET'],
                                          3,
                                          parameters['distribution'],
                                          parameters['A'],
                                          parameters['target'],
                                          parameters['mu1'],
                                          parameters['sigma1'],
                                          parameters['mu2'],
                                          parameters['sigma2'],
                                          parameters['mu3'],
                                          parameters['sigma3'],
                                          row['MIN_INTRV'],
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
                                          heu_time,
                                          str(result_downhill['allocation']),
                                          result_downhill['value'],
                                          downhill_time,
                                          result_downhill['iteration'],
                                          result_downhill['funcalls']]], columns=df_results_columns)

            df_results=df_results.append(df_run_result, ignore_index=True)
            df_results=df_results[df_results_columns]

    logging.debug('Write on disk')
    # df_results.to_excel('Results.xlsx')
    df_results.to_excel(outputfile)
    logging.debug('Program complete')

