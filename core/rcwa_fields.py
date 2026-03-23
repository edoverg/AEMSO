import meent
import numpy as np
import matplotlib.pyplot as plt

def run_fields(geom,savename):
    #=========#
    #SETUP#
    lda = 1550 #wavelength [nm]
    ucPeriod = 500 #individual unit-cell period [nm]
    ucNumber = 6 #number of unit-cells
    superCell_size = [ucPeriod*ucNumber,ucPeriod] #list of supercell [x,y] dimensions
    #MATERIALS#
    vacuum = 1
    si = 3.6388
    sio2 = 1.4518
    #GEOMETRY#
    height = 1000 #height of pillar
    #=========#
    N = 11 #Fourier expansion order
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
    result, field_cell = mee.conv_solve_field(res_z=2000, res_y=1, res_x=2000, set_field_input=(True, False, False))
    
    fig, axes = plt.subplots()
    #title = ['2D Ex', '2D Ey', '2D Ez', '2D Hx', '2D Hy', '2D Hz', ]
    title = ['2D Ex']

    ix = 0
    val = np.real(field_cell[0, :, 0, :, ix])
    im = axes.imshow(val, cmap='jet', aspect='equal')
    # plt.clim(0, 2)  # identical to caxis([-4,4]) in MATLAB
    fig.colorbar(im, ax=axes, shrink=1)
    axes.title.set_text(title[ix])

    plt.savefig("statistics/"+savename+"_result_fields.pdf",bbox_inches='tight')

    result_given_pol = result.res
    de_ri, de_ti = result_given_pol.de_ri, result_given_pol.de_ti
    print(de_ri.sum(),de_ti[N,N+1])

    return 0