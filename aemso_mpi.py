import numpy as np
from mpi4py.futures import MPIPoolExecutor
from core.rcwa_core import run_rcwa
from sklearn.preprocessing import MinMaxScaler
import cma
from core.logging_setup import logger_aemso


if __name__ == "__main__":
    #in CMA-ES, the suggested population size is: 4 + 3ln(n) = 9 for n=7
    cma_settings = {"cma_max_fun_eval":700,"pop_size":14} 

    rcwa_settings = { #define RCWA settings (materials,geometries)
        #PHYSICS PARAMETERS
        "lda" : 1550, #wavelength [nm]
        "N" : 11, #Fourier expansion order
        #MATERIALS#
        "vacuum" : 1,
        "si" : 3.6388,
        "sio2" : 1.4518,
        #GEOMETRY#
        "height" : 1000, #height of pillar
        "ucPeriod" : 500, #individual unit-cell period [nm]
        "ucNumber" : 6 #number of unit-cells
    }
    
    cma_logger = cma.CMADataLogger()
    initial_guess = 0.5*np.ones(6) #use middle value as initial guess
    es = cma.CMAEvolutionStrategy(initial_guess, 0.2, {'bounds': [0, 1],'maxfevals': cma_settings["cma_max_fun_eval"],'tolx': 1e-5,'tolfun': -0.95, 'popsize': cma_settings["pop_size"]})
    cma_logger.register(es)

    xlimits = np.array([[20, 450],[20, 450], [20, 450],[20, 450],[20, 450],[20, 450]]) #limits for unit cell's diameters
    scaler = MinMaxScaler()
    scaler.fit(xlimits.T)
    
    while not es.stop():
        #ASK: Get a population of candidates from CMA-ES
        geom_to_test_norm = es.ask(cma_settings["pop_size"])
        geom_to_test = np.rint(scaler.inverse_transform(geom_to_test_norm)) #obtain the non-normalized geometries and round
        #run simulations
        logger_aemso.info("Working on simulations...")
        rcwa_settings_ext = [rcwa_settings]*cma_settings["pop_size"]
        #spawn parallel processes to run the RCWA simulations
        #=================
        #===MPI EXECUTOR==
        #=================
        with MPIPoolExecutor() as executor:
            results_array_costfun = list(executor.starmap(run_rcwa,zip(rcwa_settings_ext,geom_to_test),chunksize=4))
        # TELL: give results back to the optimizer
        es.tell(geom_to_test_norm, results_array_costfun)
        
        cma_logger.add()
        cma_logger.save()
        msg = "Iteration " + str(es.countiter) + ": Best Y = " + str(es.best.f) + " at x = " + str(es.best.x)
        logger_aemso.info(msg)

    best_geom = np.rint(scaler.inverse_transform(np.atleast_2d(es.best.x)))
    result_fun=es.best.f
    result_geom=best_geom


    msg = "\n=========\n" + "Best obj. function= " + str(result_fun) + ", geometry= " + str(result_geom) + "\n ========="
    logger_aemso.info(msg)
    logger_aemso.info("Finished!")