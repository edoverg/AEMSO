#=
# This code performs the CMA-ES optimization in a serial way,
# meaning each geometry that is requested from the algorithm
# is tested sequentially, one after the other.
# By default, meent (through numpy) is set up to use all available logic cores=#

import cma
#from multiprocessing import Pool
import numpy as np
#import pickle
from sklearn.preprocessing import MinMaxScaler
from core.logging_setup import logger_aemso
from core.rcwa_core import run_rcwa

def optimize(cma_settings: dict,rcwa_settings: dict)->tuple:
    '''
    This function carries out the optimization. Given the settings for the CMA-ES algorithm, it returns
    the best diffraction efficiency and the associated geometry.
    
    :param cma_settings: Parameter settings for the CMA-ES algorithm 
    :type cma_settings: Dict{"cma_max_fun_eval":max number of CMA evaluations,"pop_size":CMA population size}
    :return: A tuple with the best diffraction efficiency value and associated geometry
    :rtype: tuple
    '''
    cma_logger = cma.CMADataLogger()
    #initial_guess = np.random.uniform(0,1,6) #generate a random initial guess
    initial_guess = 0.5*np.ones(6) #use middle value as initial guess
    es = cma.CMAEvolutionStrategy(initial_guess, 0.2, {'bounds': [0, 1],'maxfevals': cma_settings["cma_max_fun_eval"],'tolfun': -0.95, 'popsize': cma_settings["pop_size"]})
    cma_logger.register(es)

    xlimits = np.array([[20, 450],[20, 450], [20, 450],[20, 450],[20, 450],[20, 450]]) #limits for unit cell's diameters
    scaler = MinMaxScaler() #initialize the scaler
    scaler.fit(xlimits.T)
    
    while not es.stop():
        #es.ask: get a population of candidates from CMA-ES
        geom_to_test_norm = es.ask(cma_settings["pop_size"]) 
        #obtain the non-normalized geometries and round to nearest int
        geom_to_test = np.rint(scaler.inverse_transform(geom_to_test_norm)) 
        #run simulations
        logger_aemso.info("Working on simulations...")
        results_array_costfun = np.zeros(len(geom_to_test)) #initialize an empty array for results
        
        for i in range(len(geom_to_test)):
            #run RCWA sequentially
            results_array_costfun[i] = run_rcwa(rcwa_settings,geom_to_test[i]) 

        # TELL: give results back to the cma optimizer
        es.tell(geom_to_test_norm, results_array_costfun) 
        
        cma_logger.add()
        cma_logger.save()
        msg = "Iteration " + str(es.countiter) + ": Best Y = " + str(es.best.f) + " at x = " + str(es.best.x)
        logger_aemso.info(msg)

    best_geom = np.rint(scaler.inverse_transform(np.atleast_2d(es.best.x))) #the best (non-normalized) geometry
    return (es.best.f,best_geom)