import os
from pathlib import Path

current_dir = str(Path.cwd())
outer_dir = current_dir.split(sep="MP2")

output_file = outer_dir[0] + "flip"
print(output_file)