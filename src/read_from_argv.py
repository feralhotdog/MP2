import os 
import sys

filename = sys.argv[1]
print(filename)

fh = open(filename, 'r')
text = fh.read()

print(text)

input_file_name = os.path.basename(sys.argv[1])
name = input_file_name.split(sep=".")
print("/home/aubclsd0286/MP2/Data/results/" + name[0] + ".out")