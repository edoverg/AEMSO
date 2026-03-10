from core.logging_setup import logger_aemso
from core.optimize_serial import optimize
from core.rcwa_core import run_fields

if __name__ == "__main__":
    
    #NOTE that in CMA-ES the suggested population size is: 4 + 3ln(n) = 9 for n=6
    cma_settings = {"cma_max_fun_eval":700,"pop_size":14} #define CMA-ES settings
    
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

    result_fun,result_geom = optimize(cma_settings,rcwa_settings) #start the optimization routine
    
    msg = f"\n=========\n Best obj. function= {result_fun}, geometry= {result_geom} \n ========="
    logger_aemso.info(msg)
    
    compute_fields = False
    if compute_fields:
        logger_aemso.info("Computing fields...")    
        run_fields(result_geom[0])
    logger_aemso.info("Finished!")