import numpy as np
import matplotlib.pyplot as plt

lambdas_crown = [470, 550, 630]
indices_crown = [1.55, 1.56, 1.53]

lambda_flint = [470, 550, 630]
index_flint = [1.29, 1.29, 1.23]

plt.plot(lambdas_crown, indices_crown, 'o-', label="Crown")
plt.plot(lambda_flint, index_flint, 'o-', label="Flint")
plt.xlabel("Lambda (nm)")
plt.ylabel("Index")
plt.title("Index vs Lambda")
plt.legend()
plt.show()