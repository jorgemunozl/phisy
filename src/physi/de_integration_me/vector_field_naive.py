import numpy as np
import matplotlib.pyplot as plt


def vector_field(x, y):
    f1 = np.sin(x) + np.cos(y)
    f2 = 2*np.exp(-x**2 - y**2)
    # f3 = 3*x*y*z
    return np.array([f1, f2])

def plot_vector_field(X, Y, Z):
    plt.quiver(X, Y, Z[0], Z[1])


X = np.linspace(-2, 2, 10)
Y = np.linspace(-2, 2, 10)
X, Y = np.meshgrid(X, Y)
Z = vector_field(X, Y)
plot_vector_field(X, Y, Z)  
init_position = np.array([0, 1])
t = 0
dt = 0.1
position = np.array([0, 1])
while t < 1:
    position = position + dt * vector_field(position[0], position[1])
    t += dt
    plt.plot(position[0], position[1], 'ro')
plt.show()