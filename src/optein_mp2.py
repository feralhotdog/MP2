#===> Import Statements and Global Options <===#
import os
import sys
import psi4
import numpy as np
import time
from pathlib import Path
import tracemalloc
from opt_einsum import contract

#setting memory
psi4.set_memory(int(480e9))
numpy_memory = 480

## Input and Output File Handling
#detecting file path outside of MP2 directory
current_dir = str(Path.cwd())
outer_dir = current_dir.split(sep="MP2")

#parsing input data
filename = sys.argv[1]
fh = open(filename, 'r')
text = fh.read()

input_file_name = os.path.basename(sys.argv[1])
name = input_file_name.split(sep=".")

basis = sys.argv[2]
print(basis, name[0])

#setting output files
psi4.core.set_output_file(outer_dir[0] + "MP2/Data/results/" + name[0] + "." + basis + ".out", False)



#initiating timer and memory tracking
start_time = time.process_time_ns()
tracemalloc.start()

#===> Molecule and Psi4 Options Definitions <===#
mol = psi4.geometry(
    text
)

psi4.set_options({
    'basis':            basis,
    'scf_type':         'pk',
    'mp2_type':         'conv',
    'e_convergence':    1e-8,
    'd_convergence':    1e-8,
})

#get SCF wavefunction and energies
scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)

#===> Getting Orbital Info and energy eigenvalues <===#
#Number of Occupied orbitals and MOs
ndocc = scf_wfn.nalpha()
nmo = scf_wfn.nmo()

#get orbital energies, cast into Numpy array, separate occupied/virtual 
eps = np.asarray(scf_wfn.epsilon_a())
e_ij = eps[:ndocc]
e_ab = eps[ndocc:]

#===> ERI  <===#
#create instance of MintsHelper Class
mints = psi4.core.MintsHelper(scf_wfn.basisset())

#mem check of ERI tensor
I_size = (nmo**4) 
I_size_gb = I_size * 1e-9
print('\nSize of ERI tensor is %4.2f bytes.' % I_size)
memory_footprint = I_size * 1.5
if I_size_gb >  numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds alloted memory limit of %4.2f GB" % (memory_footprint, numpy_memory))

#build ERI tensor
I = np.asarray(mints.ao_eri())

#get MO coefficient form SCF wavefunction
C = np.asarray(scf_wfn.Ca())
Cocc = C[:, :ndocc]
Cvirt = C[:, ndocc:]

#===> Naive ERI tranformation <===#
I_mo = contract('pi,qa,pqrs,rj,sb->iajb', Cocc, Cvirt, I, Cocc, Cvirt, optimize=True)

#===> Compare I_mo with Mintshelper <===#
Co = scf_wfn.Ca_subset('AO','OCC')
Cv = scf_wfn.Ca_subset('AO', 'VIR')
MO = np.asarray(mints.mo_eri(Co, Cv, Co, Cv))
print("Do our transformed ERIs match Psi4's? %s" % np.allclose(I_mo, np.asarray(MO)))

#===> Compute MP2 Correlation & MP2 Energy <===#
#Compute energy denominator array (super naive)

# Compute energy denominator array
e_denom = 1 / (e_ij.reshape(-1, 1, 1, 1) - e_ab.reshape(-1, 1, 1) + e_ij.reshape(-1,1) - e_ab)

# Compute SS & OS MP2 Correlation with Einsum
mp2_os_corr = contract('iajb,iajb,iajb->', I_mo, I_mo, e_denom, optimize=False)
mp2_ss_corr = contract('iajb,iajb,iajb->', I_mo, I_mo - I_mo.swapaxes(1,3), e_denom)

# Total MP2 Energy
MP2_E = scf_e + mp2_os_corr + mp2_ss_corr
print("MP2 energy calculated as:" + str(MP2_E) + " AU")

#ending timer, memory tracking and reporting
end_time = time.process_time_ns()
delta_time = end_time - start_time
print("Time to complete calculation:" + str(delta_time) + " ns")

current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory usage was {peak} bytes")
tracemalloc.stop

#==> Comparing with Psi4 <==#
# ==> Compare to Psi4 <==
psi4.compare_values(psi4.energy('mp2'), MP2_E, 6, 'MP2 Energy')
print ("\n")