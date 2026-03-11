#!/bin/bash

#SBATCH --job-name AEMSO
#SBATCH --ntasks 3
#SBATCH --cpus-per-task 1
#SBATCH --partition allgroups
#SBATCH --time 00:02:00
#SBATCH --mem 8G
#SBATCH --output output_AEMSO.out
#SBATCH --error errors_AEMSO.err 
#SBATCH --nodelist=runner-01

source aemso_venv/bin/activate

export PMIX_MCA_gds=hash

mpiexec -n 3 singularity exec aemso_sing.sif python -m mpi4py aemso_mpi_2.py
