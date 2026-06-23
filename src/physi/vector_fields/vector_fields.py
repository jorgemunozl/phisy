"""
Pretty vector fields.
"""

import numpy as np
import matplotlib.pyplot as plt

def vector_field_1(x, y):
    """
    One of the simplest vector fields.
    """
    return np.array([x, y])

def vector_field_2(x, y):
    """
    One that looks like a spiral.
    """
    return np.array([-y, x])

def vector_field_3(x, y):
    """
    One that looks like an electric field between two charges.
    positive charge at the origin, negative charge at (1, 0).
    """
    pos_neg_charge = np.array([-1, 0])
    pos_charge = np.array([1, 0]) 
    return np.array([-y, x]) - pos_neg_charge / (x**2 + y**2)**(3/2) + pos_charge / ((x-1)**2 + y**2)**(3/2)

def plot_vector_field(x, y, u, v):
    plt.quiver(x, y, u, v)
    plt.show()

def main():
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    X, Y = np.meshgrid(x, y)
    U, V = vector_field_2(X, Y)
    plot_vector_field(X, Y, U, V)

if __name__ == "__main__":
    main()