import os

import numpy as np

# Path
dir_path = os.path.dirname(__file__)
files = os.listdir(dir_path + "/data")

print(files)

for file in files:
    data = np.load("data/" + file)
    print(data)
    print(np.shape(data))
