import numpy as np
import matplotlib.pyplot as plt

# First define a 2 dimensional path
def r(t: float):
    x = np.exp(t)
    y = np.sin(t)**2
    return np.array([x, y])

# Define a scalar field on the path
def f(x):
    return np.log(x[0] + x[1])

# Over all the path
N = 10000
t = np.linspace(0, 1, N)
y = r(t)
function = f(y)
dy = []

for i in range(N-1):
    dy_value = np.linalg.norm(y[:,i+1]-y[:,i])
    dy.append(dy_value)

dy = np.array(dy)
# Line Integral
line_integral = np.sum(function[:-1] * dy)
print("line_integral: ", line_integral)