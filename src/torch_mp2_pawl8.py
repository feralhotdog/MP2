#===> Import Statements and Global Options <===#
import os
import sys
import psi4
import numpy as np
import time
import torch
from pathlib import Path
import tracemalloc
from opt_einsum import contract

#setting memory
psi4.set_memory(int(480e9))
numpy_memory = 480

# Use torch?
use_torch = True

if (use_torch):
  torch.set_num_threads(4)
  torch.set_num_interop_threads(1)
  print(torch.__config__.parallel_info())
  #quit()

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




#===> Compare I_mo with Mintshelper <===#
Co = scf_wfn.Ca_subset('AO','OCC')
Cv = scf_wfn.Ca_subset('AO', 'VIR')
I = np.array(mints.ao_eri())
nbf = mints.nbf()
I = I.reshape(nbf, nbf, nbf, nbf)
if (use_torch):
  device0 = torch.device('cpu')
  I_torch = torch.tensor(I, dtype=torch.float64, device=device0)
  Co_torch = torch.tensor(np.array(Co), dtype=torch.float64, device=device0)
  Cv_torch = torch.tensor(np.array(Cv), dtype=torch.float64, device=device0)
  MO_torch = torch.einsum('pI,qA,pqrs,rJ,sB->IAJB', Co_torch, Cv_torch, I_torch, Co_torch, Cv_torch)
  MO = MO_torch.numpy()
else:
  MO = contract('pI,qA,pqrs,rJ,sB->IAJB', Co, Cv, I, Co, Cv)

#===> Compute MP2 Correlation & MP2 Energy <===#
#Compute energy denominator array (super naive)

# Compute energy denominator array
e_denom = 1 / (e_ij.reshape(-1, 1, 1, 1) - e_ab.reshape(-1, 1, 1) + e_ij.reshape(-1,1) - e_ab)

# Compute SS & OS MP2 Correlation with Einsum
MP2corr_OS = np.einsum('iajb,iajb,iajb->', MO, MO, e_denom)
MP2corr_SS = np.einsum('iajb,iajb,iajb->', MO - MO.swapaxes(1, 3), MO, e_denom)

# Total MP2 Energy
MP2_E = scf_e + MP2corr_OS + MP2corr_SS
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