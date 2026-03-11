# AEMSO
AEMSO is an Automatic electromagnetic (EM) Solver that uses Optimization algorithms to design specified periodic structures, such as beam deflectors, meta-gratings. 

AEMSO uses RCWA as EM solver, and CMA-ES as optimization algorithm.

## Features
This code uses already existing repositories. The most notable are:
+ meent: used for the RCWA implementation
+ cma: used for the CMA-ES optimization algorithm
+ mpi4py: for the MPI implementation

Apart from solving EM problems, this project aims at evaluating and profiling the computational efficiency of the algorithm. The latter, given the way it is conceived, could largely benefit from code parallelism. 

For this reason, several 'parallel code' concepts are implemented using a simplified and basic approach. We compare performance between using multithreading (exploiting the capabilities of numpy) and multiprocessing (through mpi4py).

Performance evaluations and comparisons are carried out on an HPC infrastructure, namely CLUSTER@DEI, which belongs to the Department of Information Engineering at the University of Padova.
For this reason, the code is also intended to be containerized within a Singularity (Apptainer) container.

More details about the implementation are available in the comments to the code and also in the technical report.

## Remarks
This repository is a project related to the PhD course on High Performance Computing at the University of Padova. The aim of this project is to have a very basic understanding of concepts related to HPC, and to apply them in a practical scenario.

**DISCLAIMER:** The resulting code does not necessarily implement the most efficient solution, nor it is in any case the best one. Take this project as an experiment.

## Usage
Make and move inside a new folder:
```
$ mkdir ameso_main
$ cd aemso_main
```
Next, you need to clone this repo and build the singularity image. You can do this with a simple script. Run the command:
```
$ nano setup.sh
```
then copy and paste the following code
```
#!/bin/bash
#IMPORTANT: RUN THIS SCRIPT USING INTERACTIVE MODE
echo "Building setup..."
#clone_git_repo
git clone https://github.com/edoverg/AEMSO
cd AEMSO/
#build_venv
python3 -m venv aemso_venv
#activate_venv
source aemso_venv/bin/activate
#install_requirements_in_venv
pip install openmpi
#pip install -r requirements.txt

#build_singularity
singularity build aemso_sing.sif def_file.def

echo "Setup success!"
```
save and exit the text editor. Then, activate the script:
```
$ chmod +x setup.sh
```
And finally run the script. \
**NOTE:** consider that, when building the singularity container, you might need sudo privileges. Moreover, remember to run the script within *interactive mode*
```
$ ./setup.sh
```
Once the setup is finished, you are ready to start launching the scripts.
First, move into the AEMSO folder:
```
$ cd AEMSO
```
To choose which script to launch, you need to edit the *launch_job.slurm* file with the correct filename, as is shown in the following:
```
mpiexec -n 3 singularity exec aemso_sing.sif python -m mpi4py filename.py
```
At this point one can also tweak the number of tasks and cpus-per-task to observe what happens. Once ready, you can launch the job. In slurm we do:
```
sbatch launch_job.slurm
```
