#implement MP2 calculation here. 

#===> Import Statements and Global Options <===#
import psi4
import numpy as np
from pathlib import Path

psi4.set_memory(int(2e9))
numpy_memory = 2
psi4.core.set_output_file(Path("../Data/results/'H20_MP2_out.dat'"), False)

#===> Molecule and Psi4 Options Definitions <===#
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({
    'basis':            '6-31g',
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
I_size = (nmo**4) * 8.e-9
print('\nSize of ERI tensor is %4.2f GB.' % I_size)
memory_footprint = I_size * 1.5
if I_size >  numpy_memory:
    psi4.core.clean()
    raise Exception("Estimated memory utilization (%4.2f GB) exceeds alloted memory limit of %4.2f GB" % (memory_footprint, numpy_memory))

#build ERI tensor
I = np.asarray(mints.ao_eri())

#get MO coefficient form SCF wavefunction
C = np.asarray(scf_wfn.Ca())
Cocc = C[:, :ndocc]
Cvirt = C[:, ndocc:]

#===> Naive ERI tranformation <===#
Imo = np.einsum('pi,qa,pqrs,rj,sb->iajb', Cocc, Cvirt, I, Cocc, Cvirt, optimize=True)

#===> Compare Imo with Mintshelper <===#
Co = scf_wfn.Ca_subset('AO','OCC')
Cv = scf_wfn.Ca_subset('AO', 'VIR')
MO = np.asarray(mints.mo_eri(Co, Cv, Co, Cv))
print("Do our transformed ERIs match Psi4's? %s" % np.allclose(Imo, np.asarray(MO)))

#===> Compute MP2 Correlation & MP2 Energy <===#
#Compute energy denominator array (super naive)
mp2_ss_corr = 0.0
for i in range(ndocc):
    for a in range(nmo - ndocc):
        for j in range(ndocc):
            for b in range(nmo - ndocc):
                numerator = Imo[i, a, j, b] * (Imo[i, a, j, b] - Imo[i, b, j, a])
                mp2_ss_corr += numerator / (e_ij[i] + e_ij[j] - e_ab[a] - e_ab[b])

mp2_os_corr = 0.0
for i in range(ndocc):
    for a in range(nmo - ndocc):
        for j in range(ndocc):
            for b in range(nmo - ndocc):
                numerator = Imo[i, a, j, b] * Imo[i, a, j, b]
                mp2_os_corr += numerator / (e_ij[i] + e_ij[j] - e_ab[a] - e_ab[b])

# Compute energy denominator array
e_denom = 1 / (e_ij.reshape(-1, 1, 1, 1) - e_ab.reshape(-1, 1, 1) + e_ij.reshape(-1,1) - e_ab)

# Total MP2 Energy
MP2_E = scf_e + mp2_os_corr+ mp2_ss_corr

#==> Comparing with Psi4 <==#
# ==> Compare to Psi4 <==
psi4.compare_values(psi4.energy('mp2'), MP2_E, 6, 'MP2 Energy')






