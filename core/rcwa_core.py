import meent
import numpy as np
import matplotlib.pyplot as plt
from core.logging_setup import logger_aemso
#from src.logging_setup_serial import logger_serial

def run_fields(rcwa_settings:dict,geom:np.ndarray)->int:
    '''
    Computes the EM field distribution for a given input geometry. The result is a plot that is saved on a local file
    
    :param rcwa_settings: RCWA settings, including materials and geometries
    :param geom: The supercell geometry to test
    '''
    #=========#
    #SETUP#
    lda = rcwa_settings["lda"] #wavelength [nm]
    ucPeriod = rcwa_settings["ucPeriod"] #individual unit-cell period [nm]
    ucNumber = rcwa_settings["ucNumber"] #number of unit-cells
    superCell_size = [ucPeriod*ucNumber,ucPeriod] #list of supercell [x,y] dimensions
    #theta_t = np.pi/4 #first diffraction order direction
    #MATERIALS#
    vacuum = rcwa_settings["vacuum"]
    si = rcwa_settings["si"]
    sio2 = rcwa_settings["sio2"]
    #GEOMETRY#
    height = rcwa_settings["height"] #height of pillar
    #=========#
    N = rcwa_settings["N"] #Fourier expansion order
    rcwa_options = dict(backend=0, thickness=[5*lda,height,5*lda], period=[superCell_size[0], superCell_size[1]],
                        fto=[N, N],
                        n_top=sio2, n_bot=vacuum,
                        wavelength=lda,
                        pol=1,
                        )
    
    ucell = [
        [sio2,
         []
        ],
        # layer 1
        [vacuum,
            [
                # obj 1
                ['ellipse', 0, 0, geom[0], geom[0], si, 0, 40, 40],
                ['ellipse', 1*ucPeriod, 0, geom[1], geom[1], si, 0, 40, 40],
                ['ellipse', 2*ucPeriod, 0, geom[2], geom[2], si, 0, 40, 40],
                ['ellipse', 3*ucPeriod, 0, geom[3], geom[3], si, 0, 40, 40],
                ['ellipse', 4*ucPeriod, 0, geom[4], geom[4], si, 0, 40, 40],
                ['ellipse', 5*ucPeriod, 0, geom[5], geom[5], si, 0, 40, 40],
            ],
        ],
        [vacuum,
         []
        ],
    ]

    mee = meent.call_mee(**rcwa_options)
    mee.ucell = ucell
    
    #result = mee.conv_solve()
    result, field_cell = mee.conv_solve_field(res_z=3000, res_y=1, res_x=3000, set_field_input=(True, False, False))
    
    fig, axes = plt.subplots(1,2)
    #title = ['2D Ex', '2D Ey', '2D Ez', '2D Hx', '2D Hy', '2D Hz', ]
    title = ['2D Ex']

    ix = 0
    val = np.real(field_cell[0, :, 0, :, ix])
    im = axes[ix].imshow(val, cmap='jet', aspect='auto')
    # plt.clim(0, 2)  # identical to caxis([-4,4]) in MATLAB
    fig.colorbar(im, ax=axes[ix], shrink=1)
    axes[ix].title.set_text(title[ix])

    plt.savefig("result_fields.pdf")

    result_given_pol = result.res
    de_ri, de_ti = result_given_pol.de_ri, result_given_pol.de_ti
    print(de_ri.sum(),de_ti[N,N+1])

    return 0


def run_rcwa(rcwa_settings:dict,geom_to_test:np.ndarray)->np.float64:
    '''
    Runs RCWA to compute the diffraction efficiency for a given input geometry

    :param rcwa_settings: RCWA settings, including materials and geometries
    :type rcwa_settings: Dict()    
    :param geom_to_test: The geometry to test
    :type geom_to_test: numpy.ndarray
    :return: The first order diffraction efficiency
    :rtype: np.float64
    '''
    #PHYSICS PARAMETERS
    lda = rcwa_settings["lda"] #wavelength [nm]
    N = rcwa_settings["N"] #Fourier expansion order
    #MATERIALS#
    vacuum = rcwa_settings["vacuum"]
    si   = rcwa_settings["si"]
    sio2 = rcwa_settings["sio2"]
    #GEOMETRY#
    height = rcwa_settings["height"] #height of pillar
    ucPeriod = rcwa_settings["ucPeriod"] #individual unit-cell period [nm]
    ucNumber = rcwa_settings["ucNumber"] #number of unit-cells
    superCell_size = [ucPeriod*ucNumber,ucPeriod] #list of supercell [x,y] dimensions
    #=========#
    
    rcwa_options = dict(backend=0, thickness=[5*lda,height,5*lda], period=[superCell_size[0], superCell_size[1]],
                        fto=[N, N],
                        n_top=sio2, n_bot=1,
                        wavelength=lda,
                        pol=1,
                        )
    ucell = [
        [sio2,
        []
        ],
        # layer 1
        [vacuum,
            [
                # obj 1
                ['ellipse', 0, 0, geom_to_test[0], geom_to_test[0], si, 0, 40, 40],
                ['ellipse', 1*ucPeriod, 0, geom_to_test[1], geom_to_test[1], si, 0, 40, 40],
                ['ellipse', 2*ucPeriod, 0, geom_to_test[2], geom_to_test[2], si, 0, 40, 40],
                ['ellipse', 3*ucPeriod, 0, geom_to_test[3], geom_to_test[3], si, 0, 40, 40],
                ['ellipse', 4*ucPeriod, 0, geom_to_test[4], geom_to_test[4], si, 0, 40, 40],
                ['ellipse', 5*ucPeriod, 0, geom_to_test[5], geom_to_test[5], si, 0, 40, 40],
            ],
        ],
        [vacuum,
        []
        ],
    ]

    mee = meent.call_mee(**rcwa_options)
    mee.ucell = ucell
    result = mee.conv_solve()

    result_given_pol = result.res
    #de_ri = result_given_pol.de_ri #get reflection orders
    de_ti = result_given_pol.de_ti #get transmission orders

    # NOTE: de_ri and de_ti are arrays of shape [2N+1,2N+1]. 
    # The zero order is at index N, and the +1 diff order is at 
    # index N+1
    logger_aemso.info("RCWA eval:" + str(geom_to_test) + " | " + "diff. eff.:" + str(de_ti[N,N+1]))
    return -de_ti[N,N+1] #return the negative, since we want to maximize the diff. efficiency