import numpy as np
from core.rcwa_fields import run_fields
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


#limits for unit cell's diameters
xlimits = np.array([[20, 450],[20, 450], [20, 450],[20, 450],[20, 450],[20, 450]]) 
#initialize the scaler
scaler = MinMaxScaler() 
scaler.fit(xlimits.T)

def make_fields(data_dict):
    for it in data_dict.items():
        savename = it[0]
        geom_to_test_norm = it[1]    
        
        geom_int = np.rint(scaler.inverse_transform(geom_to_test_norm.reshape(1,-1))) 
        run_fields(geom_int[0],savename)

if __name__ == '__main__':
    #optimized geometries results
    input_data_dict = {
                       "aemso_serial_logs_ameso_ncpu_1":np.atleast_2d([0.81498735,0.1451117 ,0.46977402,0.60117747,0.67328409,0.74997751]),
                       "aemso_serial_logs_ameso_ncpu_2":np.atleast_2d([0.60534464,0.69459403,0.75956666,0.81235771,0.02759088,0.45437877],),
                       "aemso_serial_logs_ameso_ncpu_4":np.atleast_2d([0.24634709,0.47632482,0.60643513,0.67699909,0.74828158,0.81126211]),
                       "aemso_serial_logs_ameso_ncpu_8":np.atleast_2d([0.549866643,0.649700972,0.719981111,0.748572037,0.371782339e-04,2.14502275e-03]),
                       "aemso_serial_logs_ameso_ncpu_14":np.atleast_2d([0.00394863,0.51755722,0.99969483,0.68355503,0.71700614,0.80145616]),
                       "aemso_mpi_2p_ncpu_2":np.atleast_2d([0.67584711,0.74800172,0.81526742,0.10628601,0.48164934,0.60539626]),
                       "aemso_mpi_2p_ncpu_7":np.atleast_2d([0.66674982, 0.73899593, 0.80719075, 0.04802076, 0.4589818,  0.59910317]),
                       "aemso_mpi_2p_ncpu_14":np.atleast_2d([0.21212098, 0.2080828,  0.58834569, 0.65552659, 0.73817846, 0.79286903])
                       }

    #plot optimized geometries
    plt.figure(1)
    ucell_index = [1,2,3,4,5,6]
    serial1 = np.sort(np.rint(scaler.inverse_transform(input_data_dict["aemso_serial_logs_ameso_ncpu_1"].reshape(1,-1))))
    serial2 = np.sort(np.rint(scaler.inverse_transform(input_data_dict["aemso_serial_logs_ameso_ncpu_2"].reshape(1,-1))))
    serial4 = np.sort(np.rint(scaler.inverse_transform(input_data_dict["aemso_serial_logs_ameso_ncpu_4"].reshape(1,-1))))
    serial8 = np.sort(np.rint(scaler.inverse_transform(input_data_dict["aemso_serial_logs_ameso_ncpu_8"].reshape(1,-1))))
    serial14 = np.sort(np.rint(scaler.inverse_transform(input_data_dict["aemso_serial_logs_ameso_ncpu_14"].reshape(1,-1))))
    mpi2 = np.sort(np.rint(scaler.inverse_transform(input_data_dict["aemso_mpi_2p_ncpu_2"].reshape(1,-1))))
    mpi7 = np.sort(np.rint(scaler.inverse_transform(input_data_dict["aemso_mpi_2p_ncpu_7"].reshape(1,-1))))
    mpi14 = np.sort(np.rint(scaler.inverse_transform(input_data_dict["aemso_mpi_2p_ncpu_14"].reshape(1,-1))))
    plt.plot(ucell_index,*serial1,marker="o")
    plt.plot(ucell_index,*serial2,marker="o")
    plt.plot(ucell_index,*serial4,marker="o")
    plt.plot(ucell_index,*serial8,marker="o")
    plt.plot(ucell_index,*serial14,marker="o")
    plt.plot(ucell_index,*mpi2,marker="o")
    plt.plot(ucell_index,*mpi7,marker="o")
    plt.plot(ucell_index,*mpi14,marker="o")
    plt.xlabel("Unit cell",fontsize=14)
    plt.ylabel("Size [nm]",fontsize=14)
    plt.legend(["Serial - 1","Serial - 2","Serial - 4","Serial - 8","Serial - 14","MPI - 2","MPI - 7","MPI - 14"],ncol=4,fontsize=10)
    plt.tight_layout()
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("statistics/opt_geoms.pdf")

    #make_fields(input_data_dict)
    
    # PERFORMANCE ANALYSIS
    #this data was extracted using the seff command, and can be found in the log files
    mpi2p_ncpus = [2,7,14]
    time_stat_mpi2p = [2.32,0.7,0.35]
    cpu_stat_mpi2p = [0.99,0.99,0.99]
    mem_stat_mpi2p = [2,6.9,13.5]
    
    serial_ncpus = [1,2,4,8,14]
    time_stat_serial = [4.59,2.65,1.75,1.37,1.42]
    mem_stat_serial = [1,1,1,1,1]
    cpu_stat_serial = [1.01,1.01,1.01,1.02,1.04]
    
    plt.figure(2)
    plt.plot(serial_ncpus,time_stat_serial,marker="^",c="b")
    plt.plot(mpi2p_ncpus,time_stat_mpi2p,marker='d',c='g')
    plt.xlabel("Number of tasks-CPUs",fontsize=14)
    plt.ylabel("Execution time [h]",fontsize=14)
    plt.legend(["Serial","MPI Scatter"],fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.savefig("statistics/cpuVtime.pdf")

    plt.figure(3)

    plt.plot(serial_ncpus,mem_stat_serial,marker='^',c='b')
    plt.plot(mpi2p_ncpus,mem_stat_mpi2p,marker='d',c='g')
    plt.xlabel("Number of tasks-CPUs",fontsize=14)
    plt.ylabel("Memory usage [GB]", fontsize=14)
    plt.legend(["Serial","MPI Scatter"])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig("statistics/cpuVmem.pdf")

    plt.figure(4)

    plt.plot(serial_ncpus,cpu_stat_serial,marker="^",c="b")
    plt.plot(mpi2p_ncpus,cpu_stat_mpi2p,marker='d',c='g')
    plt.xlabel("Number of tasks-CPUs",fontsize=14)
    plt.ylabel("CPU efficiency", fontsize=14)
    plt.legend(["Serial","MPI Scatter"])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim((0,1))

    plt.savefig("statistics/cpuEff.pdf")