import numpy as np
import matplotlib.pyplot as plt


B = np.array([0.9, 1.11, 1.44, 1.59, 1.84, 2.02, 2.30, 2.51, 2.75, 3.05, 3.29, 3.54, 3.77, 4.85])
I = np.array([0.47, 0.57, 0.66, 0.85, 0.97, 1.07, 1.21, 1.34, 1.47, 1.64, 1.76 , 1.89, 2.02, 2.61])

plt.scatter(
    I,
    B,
    colorizer="red"
)
plt.xlabel("I (A)")
plt.ylabel("B (mT)")
plt.title("Variacion Intensidad - Magnitud campo ")
plt.savefig("plot", dpi=500)