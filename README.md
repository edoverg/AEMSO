# AEMSO
AEMSO is an Automatic electromagnetic (EM) Solver that uses Optimization algorithms to design specified periodic structures, such as beam deflectors, meta-gratings. 

AEMSO uses RCWA as EM solver, and CMA-ES as optimization algorithm.

## Features
This code uses already existing repositories:
+ meent: used for the RCWA implementation
+ cma: used for the CMA-ES optimization algorithm

Apart from solving EM problems, this project has a great interest in evaluating performance from a computational point of view. For how it is conceived, the code can benefit from parallelism. 

For this reason, several implementation of parallelism concepts are implemented using a simplified and basic approach. The compares performance using multithreading (exploiting the capabilities of numpy) and multiprocessing (though mpi4py).

Performance evaluations and comparisons are carried out on an HPC infrastructure, namely CLUSTER@DEI, which belongs to the Department of Information Engineering at the University of Padova.
For this reason, the code is also inteded to be containerized within a Singularity (Apptainer) container.

More implementations details are available in the technical report.

## Remarks
This repository is a project related to the PhD course on High Performance Computing at the University of Padova. The aim of this project is to have a very basic understanding of concepts related to HPC, and to apply them to a practical case of interest.