from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad


def f(x):
    return x


def obtain_coeff(f: Callable, n: int):
    a_n = np.zeros((n))
    b_n = np.zeros((n))
    for i in range(n):
        a_n[i] = quad(lambda x: f(x) * np.cos(i * x), -np.pi, np.pi)[0]
        b_n[i] = quad(lambda x: f(x) * np.sin(i * x), -np.pi, np.pi)[0]
    return a_n, b_n


def fourier_series(a_0, a_n, b_n, n: int):
    i_n = a_0
    for i in range(n):
        i_n += a_n[i] * np.cos(i * n) + b_n[i] * np.sin(i * n)
    return i_n


def main():
    x = np.linspace(-np.pi, np.pi)
    y = f(x)

    a_n, b_n = obtain_coeff(f, 10)
    y_prime = fourier_series(0, a_n, b_n, 10)
    plt.plot(x, y)
    plt.plot(x, y_prime)
    plt.show()
    print("FINISHED")


if __name__ == "__main__":
    main()
