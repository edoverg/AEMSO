from mpi4py import MPI
import numpy as np
import cma
from sklearn.preprocessing import MinMaxScaler
from core.logging_setup import logger_aemso
from core.rcwa_core import run_rcwa
import itertools

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data = None

es = None
flag = False

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
#rcwa_settings_ext = [rcwa_settings]*size
cma_settings = {"cma_max_fun_eval":5,"pop_size":5} #define CMA-ES settings
#setup the CMA instance
es = None
if rank == 0:
    cma_logger = cma.CMADataLogger()
        #initial_guess = np.random.uniform(0,1,6) #generate a random initial guess
    initial_guess = 0.5*np.ones(6) #use middle value as initial guess
    es = cma.CMAEvolutionStrategy(initial_guess, 0.2, {'bounds': [0, 1],'maxfevals': cma_settings["cma_max_fun_eval"],'tolfun': -0.95, 'popsize': cma_settings["pop_size"]})
    cma_logger.register(es)

geom_to_test_chunks = None
while not flag:
    print(f"rank {rank} inside while")
    if rank == 0:        
        xlimits = np.array([[20, 450],[20, 450], [20, 450],[20, 450],[20, 450],[20, 450]]) #limits for unit cell's diameters
        scaler = MinMaxScaler() #initialize the scaler
        scaler.fit(xlimits.T)
        geom_to_test_norm = es.ask(cma_settings["pop_size"]) 
        #obtain the non-normalized geometries and round to nearest int
        geom_to_test = np.rint(scaler.inverse_transform(geom_to_test_norm)) 
        #split data into chunks
        geom_to_test_chunks = np.array_split(geom_to_test,size)
        #print(f"GEOM_TO_TEST_CHUNKS {geom_to_test_chunks}")
        geom_to_test_chunks = [list(c) for c in geom_to_test_chunks]
        #print(f"geom to test after list = {geom_to_test_chunks}")
        #run simulations    
        logger_aemso.info("Working on simulations...")

        results_array_costfun = np.zeros(len(geom_to_test)) #initialize an empty array for results
    
    #print(f"geom_to_test_chunks rank {rank} = {geom_to_test_chunks}")        
    test_this_geom = comm.scatter(geom_to_test_chunks,root = 0)
    #print(f"Rank {rank} got this geom {test_this_geom}")
    results_array_costfun = [run_rcwa(rcwa_settings,geom) for geom in test_this_geom] 
    all_results_array_costfun = comm.gather(results_array_costfun,root=0)

    if rank == 0:
        #print(f"Gathered results = {list(itertools.chain.from_iterable(all_results_array_costfun))}")
        #print(f"geom_to_test_norm = {geom_to_test_norm}")
        #TELL: give results back to the cma optimizer
        es.tell(geom_to_test_norm, list(itertools.chain.from_iterable(all_results_array_costfun))) 
        cma_logger.add()
        cma_logger.save()
        flag = es.stop() #check if the stop criterion is met
        msg = "Iteration " + str(es.countiter) + ": Best Y = " + str(es.best.f) + " at x = " + str(es.best.x)
        logger_aemso.info(msg)
        #print("Flag = ",flag)
    #print(f"Rank {rank} found a barrier")
    comm.Barrier()
    flag = comm.bcast(flag,root=0) #send the updated flag to all workers
    #print(f"Rank {rank} sees a flag {flag}")
    #print("Exiting barrier")