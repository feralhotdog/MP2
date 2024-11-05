#!/bin/bash

source ~/.bashrc.local.asax
conda activate mp2

for i in $(readlink -f /home/aubclsd0286/MP2/Data/input_files/*) 
do
	python3 /home/aubclsd0286/MP2/src/naive_mp2.py $i cc-pVQZ
done

conda deactivate
exit 0