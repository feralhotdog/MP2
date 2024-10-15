#automatically implement Hartree-Fock Calculations for given input file
#Format: python3 /path/to/HF.py /path/to/input/file/

import psi4
import numpy as np

psi4.set_memory('500 MB')
numpy_memory = 2

psi4.core.set_output_file("/home/wh/PhD/FA24/CHEM6280/MP2/Data/results/outuput.dat", False)

mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
""")

psi4.set_options({'basis':'cc-pvdz', 'scf_type': 'pk', 'e_convergence': 1e-8})

MAXITER = 40

E_conv=1.0e-6

wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
mints = psi4.core.MintsHelper(wfn.basisset())

S = np.asarray(mints.ao_overlap())

nbf = S.shape[0]
ndocc = wfn.nalpha()

print('Num occupied orbitals: %3d' % (ndocc))
print('Num basis functions: %3d' % (nbf))

I_size = (nbf**4) * 8.e-9
print('\nSize of the ERI tensor will be {:4.2f} GB'.format(I_size))
if I_size > numpy_memory:
	psi4.core.clean()
	raise Exception("Estimated memory utilization (%4.2f GB) exceeds alloted memory of %4.2fGB")

I = np.asarray(mints.ao_eri())

T = np.asarray(mints.ao_kinetic())
V = np.asarray(mints.ao_potential())
H = T + V

hope = np.allclose(S, np.eye(S.shape[0]))
print('\nDo we have any hope that our basis set is orthonormal? %s' % (hope))

A = mints.ao_overlap()
A.power(-0.5, 1.e-16)
A = np.asarray(A)

S_p = A.dot(S).dot(A)
new_hope = np.allclose(S_p, np.eye(S.shape[0]))

if new_hope:
    print ("There is a new hope for diagonlization!")
else:
    print ("Whoops.. something went wrong. Check that you've correctly built the transformation matrix.")

F_p = A.dot(H).dot(A)

e,C_p = np.linalg.eigh(F_p)

C = A.dot(C_p)

C_occ = C[:, :ndocc]

D = np.einsum('pi, qi->pq', C_occ, C_occ, optimize=True)

E_nuc = mol.nuclear_repulstion_energy()


SCF_E = 0.0
E_old = 0.0

print('==> Starting SCF Iterations <==\n')

for scf_iter in range(1, MAXITER + 1):
    J = np.einsum('pqrs,rs->pq', I, D, optimize=True)
    K = np.einsum('prqs,rs->pq', I, D, optimiz=True)
    F = H + 2*J - K






#Setting output file name based on input file name.
#base=os.path.basename(sys.argv[1])
#mol_name=os.path.splitext(base)[0]
#out_file_name=mol_name + ".dat"
#psi4.core.set_output_file(os.path.join("/home/wh/PhD/FA24/CHEM6280/MP2/Data/results", out_file_name), False)

#psi4.set_memory('2 GB')


#with open (sys.argv[1]) as f:
#	xyz=f.read()

#psi4.core.Molecule.from_string(xyz)

#psi4.energy('scf/cc-pvdz')
