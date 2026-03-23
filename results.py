import numpy as np
from core.rcwa_fields import run_fields
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

xlimits = np.array([[20, 450],[20, 450], [20, 450],[20, 450],[20, 450],[20, 450]]) #limits for unit cell's diameters
scaler = MinMaxScaler() #initialize the scaler
scaler.fit(xlimits.T)


def make_fields(data_dict):
    #geom_to_test_norm = np.atleast_2d([0.17081771,0.46180442,0.60250453,0.67680841,0.7471375,0.81101642],)
    for it in data_dict.items():
        savename = it[0]
        geom_to_test_norm = it[1]    
        
        geom_int = np.rint(scaler.inverse_transform(geom_to_test_norm.reshape(1,-1))) 
        run_fields(geom_int[0],savename)


if __name__ == '__main__':
    #optimized geometries results
    input_data_dict = {"aemso_mpi_2_ncpu_2":np.atleast_2d([0.17081771,0.46180442,0.60250453,0.67680841,0.7471375,0.81101642]),
                       "aemso_mpi_2_ncpu_4":np.atleast_2d([0.66011131,0.74738171,0.80390325,0.3014312 ,0.35140446,0.59872171]),
                       "aemso_mpi_2_ncpu_8":np.atleast_2d([0.67446952,0.74726075,0.80926722,0.04021622,0.46275808,0.60193284]),
                       "aemso_mpi_2_ncpu_14":np.atleast_2d([0.74517123,0.80699085,0.20801253,0.45360801,0.60366698,0.66376855]),
                       "aemso_serial_logs_ameso_ncpu_1":np.atleast_2d([0.81498735,0.1451117 ,0.46977402,0.60117747,0.67328409,0.74997751]),
                       "aemso_serial_logs_ameso_ncpu_2":np.atleast_2d([0.60534464,0.69459403,0.75956666,0.81235771,0.02759088,0.45437877],),
                       "aemso_serial_logs_ameso_ncpu_4":np.atleast_2d([0.24634709,0.47632482,0.60643513,0.67699909,0.74828158,0.81126211]),
                       "aemso_serial_logs_ameso_ncpu_8":np.atleast_2d([0.549866643,0.649700972,0.719981111,0.748572037,0.371782339e-04,2.14502275e-03]),
                       "aemso_serial_logs_ameso_ncpu_14":np.atleast_2d([0.00394863,0.51755722,0.99969483,0.68355503,0.71700614,0.80145616]),
                       "aemso_mpi_2p_ncpu_2":np.atleast_2d([0.67584711,0.74800172,0.81526742,0.10628601,0.48164934,0.60539626]),
                       "aemso_mpi_2p_ncpu_7":np.atleast_2d([0.66674982, 0.73899593, 0.80719075, 0.04802076, 0.4589818,  0.59910317]),
                       "aemso_mpi_2p_ncpu_14":np.atleast_2d([0.21212098, 0.2080828,  0.58834569, 0.65552659, 0.73817846, 0.79286903])
                       }

    #make_fields(input_data_dict)
    
    # PERFORMANCE ANALYSIS
    #this data was extracted using the seff command, and can be found in the log files
    mpi2_ncpus = [2,4,8,14]
    time_stat_mpi2 = [4.5,2.68,1,0.65]
    mem_stat_mpi2 = [2,4,7.9,13.6]
    cpu_stat_mpi2 = [0.99,0.99,0.99,0.99]

    mpi2p_ncpus = [2,7,14]
    time_stat_mpi2p = [2.32,0.7,0.35]
    cpu_stat_mpi2p = [0.99,0.99,0.99]
    mem_stat_mpi2p = [2,6.9,13.5]

    mpiPool_ncpus = [4,8,14]
    time_stat_mpiPool = [3,1.1,0.65]
    mem_stat_mpiPool = [3.3,7,11]
    cpu_stat_mpiPool = [0.6,0.86,0.53]
    
    serial_ncpus = [1,2,4,8,14]
    time_stat_serial = [4.59,2.65,1.75,1.37,1.42]
    mem_stat_serial = [1,1,1,1,1]
    cpu_stat_serial = [0.99,0.99,0.98,0.98,0.96]
    
    plt.figure(1)
    plt.plot(serial_ncpus,time_stat_serial,marker="^",c="b")
    plt.plot(mpi2_ncpus,time_stat_mpi2,marker="o",c="r")
    plt.plot(mpi2p_ncpus,time_stat_mpi2p,marker='d',c='g')
    plt.plot(mpiPool_ncpus,time_stat_mpiPool,marker="*",c='k')
    plt.xlabel("Number of tasks-CPUs",fontsize=14)
    plt.ylabel("Execution time [h]",fontsize=14)
    plt.legend(["Serial","MPI scatter","MPI Scatter","MPI pool"],fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.savefig("statistics/cpuVtime.pdf")

    plt.figure(2)

    plt.plot(serial_ncpus,mem_stat_serial,marker='^',c='b')
    plt.plot(mpi2_ncpus,mem_stat_mpi2,marker='o',c='r')
    plt.plot(mpi2p_ncpus,mem_stat_mpi2p,marker='d',c='g')
    plt.plot(mpiPool_ncpus,mem_stat_mpiPool,marker='*',c='k')
    plt.xlabel("Number of tasks-CPUs",fontsize=14)
    plt.ylabel("Memory usage [GB]", fontsize=14)
    plt.legend(["Serial","MPI scatter","MPI Scatter","MPI pool"])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig("statistics/cpuVmem.pdf")

    plt.figure(3)

    plt.plot(serial_ncpus,cpu_stat_serial,marker="^",c="b")
    plt.plot(mpiPool_ncpus,cpu_stat_mpiPool,marker="*",c='k')
    plt.plot(mpi2_ncpus,cpu_stat_mpi2,marker="o",c="r")
    plt.plot(mpi2p_ncpus,cpu_stat_mpi2p,marker='d',c='g')
    plt.xlabel("Number of tasks-CPUs",fontsize=14)
    plt.ylabel("CPU efficiency", fontsize=14)
    plt.legend(["Serial","MPI Pool","MPI scatter","MPI Scatter"])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim((0,1))

    plt.savefig("statistics/cpuEff.pdf")