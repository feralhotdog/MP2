#automatically implement Hartree-Fock Calculations for given input file
#Format: python3 /path/to/HF.py /path/to/input/file/

import psi4
import os
import sys


#Setting output file name based on input file name.
base=os.path.basename(sys.argv[1])
mol_name=os.path.splitext(base)[0]
out_file_name=mol_name + ".dat"
psi4.core.set_output_file(os.path.join("/home/wh/PhD/FA24/CHEM6280/MP2/Data/results", out_file_name), False)

psi4.set_memory('2 GB')


with open (sys.argv[1]) as f:
	xyz=f.read()

psi4.core.Molecule.from_string(xyz)

psi4.energy('scf/cc-pvdz')