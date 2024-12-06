# Naive Implementation of Moller-Plesset perturbation theory of the second order.

## Data To Track for Calculations
1. Energy and accuracy (compared to psi4 built in)
2. Time (cputime/runtime)
3. Memory Usage (active amount vs total)

## Check List / Notes
 - Molecular Data pulled from pubchem in .sdf format. Converted to .gzmat with Avogadro.
 - Naive implementation of MP2 derived from https://github.com/psi4/psi4numpy/blob/master/Tutorials/05_Moller-Plesset/5a_conventional-mp2.ipynb, with alterations to allow for reading input files.
 - Make utility for tracking CPU Time?
 - Make utility for tracking memory usage.
 - Run base level calculations
 - Libraries for ML utilized to parallelize tensor math.
 - Reimplementation of code in Bend language (To do)
 - Format memory to read only pertinent integrals as needed (To Do)
