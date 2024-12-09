#!/bin/bash

source ~/.bashrc.local.asax
conda activate mp2

$MYSCRATCH="/home/aubclsd0286/Scratch"

basis_set="6-31g cc-PVDZ cc-PVTZ cc-PVQZ"

for basis in $(echo $basis_set)
do
	for i in $(readlink -f /home/aubclsd0286/MP2/Data/input_files/*) 
	do
		python3 /home/aubclsd0286/MP2/src/torch_mp2_psi4_thr.py $i $basis
	done
done

conda deactivate
exit 0