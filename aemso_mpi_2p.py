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
cma_settings = {"cma_max_fun_eval":700,"pop_size":14} #define CMA-ES settings

#setup the CMA instance
es = None
if rank == 0:
    cma_logger = cma.CMADataLogger()
    initial_guess = 0.5*np.ones(6) #use half-value as initial guess
    es = cma.CMAEvolutionStrategy(initial_guess, 0.2, {'bounds': [0, 1],'maxfevals': cma_settings["cma_max_fun_eval"],
                                                       'tolfun': -0.95, 'popsize': cma_settings["pop_size"]})
    cma_logger.register(es)

geom_to_test_flat = None
while not flag:
    if rank == 0:       
        #limits for unit cell's diameters 
        xlimits = np.array([[20, 450],[20, 450], [20, 450],[20, 450],[20, 450],[20, 450]]) 
        scaler = MinMaxScaler() #initialize the scaler
        scaler.fit(xlimits.T)
        geom_to_test_norm = es.ask(cma_settings["pop_size"]) 
        #obtain the non-normalized geometries and round to nearest int
        geom_to_test = np.rint(scaler.inverse_transform(geom_to_test_norm))
        geom_to_test_flat = np.rint(scaler.inverse_transform(geom_to_test_norm)).astype(np.int32).flatten()
        #split data into chunks
        index_worker = np.int16(geom_to_test_flat.size/size)
        counts = np.array([index_worker]*size,dtype='i')
        displacements = np.arange(0,geom_to_test_flat.size,index_worker,dtype='i')
        #run simulations    
        logger_aemso.info("Working on simulations...")

        results_array_costfun = np.zeros(len(geom_to_test)) #initialize an empty array for results
    else:
        counts = None
        displacements = None

    recv_buf = np.empty(np.int32(cma_settings["pop_size"]*rcwa_settings["ucNumber"]/size),dtype='i')

    comm.Scatterv([geom_to_test_flat,(counts,displacements),MPI.INT],recv_buf,root = 0)
    #reshape array
    test_this_geom = recv_buf.reshape(-1,rcwa_settings["ucNumber"])
    results_array_costfun = np.array([run_rcwa(rcwa_settings,geom) for geom in test_this_geom])
    all_results_array_costfun = np.empty(cma_settings["pop_size"])
    
    comm.Gather(results_array_costfun,all_results_array_costfun,root=0)

    if rank == 0:
        #TELL: give results back to the cma optimizer
        es.tell(geom_to_test_norm, all_results_array_costfun)
        cma_logger.add()
        cma_logger.save()
        flag = es.stop() #check if the stop criterion is met
        msg = "Iteration " + str(es.countiter) + ": Best Y = " + str(es.best.f) + " at x = " + str(es.best.x)
        logger_aemso.info(msg)
        
    comm.Barrier()
    flag = comm.bcast(flag,root=0) #send the updated flag to all workers