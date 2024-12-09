# ==> Import statements & Global Options <==
import os
import sys
import psi4
import numpy as np
import time
from pathlib import Path
import tracemalloc
from dataclasses import dataclass

@dataclass
class datapoint:
    value1: str
    value2: float
    value3: float

    def __str__(self):
        return f"{self.value1}, {self.value2}, {self.value3}"

def mem_time(label): #saves peak memory and comp time to dictionary
    global start_time
    global mem_time_dict

    calc_time = time.process_time_ns() - start_time
    start_time += calc_time

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    m_t_data.append(datapoint(label, calc_time, peak))
    

#starting memory and time characterization
tracemalloc.start()
m_t_data = []
start_time = time.process_time_ns()

mem_time("initial") #baseline memory for initializing and base time

#setting memory/threading
psi4.set_num_threads(1)
psi4.set_memory(int(2e9))
numpy_memory = 2

mem_time("mem_alloc") #memory allocation and thread management



#Input and Output File setup
current_dir = str(Path.cwd())
outer_dir = current_dir.split(sep="MP2")

#reading input data
input_file = sys.argv[1]
handle = open(input_file, 'r')
text = handle.read()
mol_name = os.path.basename(sys.argv[1]).split(sep=".")
basis = sys.argv[2]

print(basis, mol_name[0])


#setting output files
psi4.core.set_output_file(outer_dir[0] + "MP2/Data/results/" + mol_name[0] + "." + basis + ".out", False)

mem_time("I/O")#input and output handling



#===> Molecule and Psi4 Options Definitions <===#
mol = psi4.geometry(
    text
)

psi4.set_options({'basis':        basis,
                  'scf_type':     'pk',
                  'mp2_type':     'conv',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

mem_time("geo_opt") #Psi4 Geometry and options



# Get the SCF wavefunction & energies
scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)

mem_time("psi4_wfn") #Psi4 wavefunction/energies

# ==> Get orbital information & energy eigenvalues <==
# Number of Occupied orbitals & MOs
ndocc = scf_wfn.nalpha()
nmo = scf_wfn.nmo()

mem_time("Orb_info") #getting orbital/energy


# Get orbital energies, cast into NumPy array, and separate occupied & virtual
eps = np.asarray(scf_wfn.epsilon_a())
e_ij = eps[:ndocc]
e_ab = eps[ndocc:]

mem_time("orb_manip") #Numpy array manipulation


# ==> ERIs <==
# Create instance of MintsHelper class
mints = psi4.core.MintsHelper(scf_wfn.basisset())

# Memory check for ERI tensor
I_size = (nmo**4) * 8.e-9
print('\nSize of the ERI tensor will be %4.2f GB.' % I_size)
memory_footprint = I_size * 1.5
if I_size > numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted memory \
                     limit of %4.2f GB." % (memory_footprint, numpy_memory))

mem_time("mem_check") #Check ERI tensor size



# Build ERI Tensor
I = np.asarray(mints.ao_eri())

mem_time("ERI_build") #build ERI tensor


# Get MO coefficients from SCF wavefunction
C = np.asarray(scf_wfn.Ca())
Cocc = C[:, :ndocc]
Cvirt = C[:, ndocc:]

mem_time("coefficients") #track time for coefficient grabbn


# ==> Transform I -> I_mo @ O(N^5) <==
tmp = np.einsum('pi,pqrs->iqrs', Cocc, I, optimize=True)
tmp = np.einsum('qa,iqrs->iars', Cvirt, tmp, optimize=True)
tmp = np.einsum('iars,rj->iajs', tmp, Cocc, optimize=True)
I_mo = np.einsum('iajs,sb->iajb', tmp, Cvirt, optimize=True)

mem_time("transformation") #AO to MO transformation


# ==> Compare our Imo to MintsHelper <==
Co = scf_wfn.Ca_subset('AO','OCC')
Cv = scf_wfn.Ca_subset('AO','VIR')
MO = np.asarray(mints.mo_eri(Co, Cv, Co, Cv))
print("Do our transformed ERIs match Psi4's? %s" % np.allclose(I_mo, np.asarray(MO)))

mem_time("ImoCompare") #compares to psi4 Imo


# ==> Compute MP2 Correlation & MP2 Energy <==
# Compute energy denominator array
e_denom = 1 / (e_ij.reshape(-1, 1, 1, 1) - e_ab.reshape(-1, 1, 1) + e_ij.reshape(-1, 1) - e_ab)

mem_time("e_denom") #finding energy denominator


# Compute SS & OS MP2 Correlation with Einsum
mp2_os_corr = np.einsum('iajb,iajb,iajb->', I_mo, I_mo, e_denom, optimize=True)
mp2_ss_corr = np.einsum('iajb,iajb,iajb->', I_mo, I_mo - I_mo.swapaxes(1,3), e_denom, optimize=True)

mem_time("numerator") #calculating numerator


# Total MP2 Energy
MP2_E = scf_e + mp2_os_corr + mp2_ss_corr

mem_time("final_energy") #calculate total energy


# ==> Compare to Psi4 <==
psi4.compare_values(psi4.energy('mp2'), MP2_E, 6, 'MP2 Energy')

mem_time("final_compare") #final comparison


for dp in m_t_data:
    print(dp)