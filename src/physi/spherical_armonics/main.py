import matplotlib.pyplot as plt
import numpy as np
import math
from utils import double_fac


class Legendre:
    """
    Simple Legendre polynomial generator based on the recurrence
    P_n(x) = ((2n-1)x P_{n-1}(x) - (n-1) P_{n-2}(x)) / n.
    """

    def __init__(self, n: int) -> None:
        if n < 0:
            raise ValueError("Need n >= 0")
        self.n = n

    def polynomial(self, x):
        """
        Evaluate P_n(x) using the standard recursion.
        """
        x_arr = np.asarray(x, dtype=float)
        if self.n == 0:
            return np.ones_like(x_arr, dtype=float)
        if self.n == 1:
            return x_arr

        p_nm2 = np.ones_like(x_arr, dtype=float)  # P_0
        p_nm1 = x_arr  # P_1
        for ell in range(2, self.n + 1):
            p_n = ((2 * ell - 1) * x_arr * p_nm1 - (ell - 1) * p_nm2) / ell
            p_nm2, p_nm1 = p_nm1, p_n
        return p_nm1

    def pol_2(self, x):
        """
        Comparation
        """
        return 1/2*(3*np.pow(x, 2)-1)


def legendre_5(x):
    return 1/8*(63*np.pow(x, 5)-70*np.pow(x, 3)+15*x)


def legendre_derivative(n, x):
    """
    Returns the derivative of the n legendre function
    """
    diff = legendre_polinomial(n-1, x) - x*legendre_polinomial(n, x)
    return n/(1-np.pow(x, 2))*diff


def P_lm(l, m, x):
    """
    Associated Legendre P_l^m(x) for 0 <= m <= l
    using standard recurrences.
    """
    if m < 0 or m > l:
        raise ValueError("Need 0 <= m <= l")

    # P_m^m(x)
    P_mm = (-1)**m * double_fac(2*m - 1) * (1 - x**2)**(m/2)
    if l == m:
        return P_mm

    # P_{m+1}^m(x)
    P_m1m = x * (2*m + 1) * P_mm
    if l == m + 1:
        return P_m1m

    # Upward recurrence in l
    P_lm2 = P_mm   # P_m^m
    P_lm1 = P_m1m  # P_{m+1}^m

    for ell in range(m + 2, l + 1):
        P_l = ((2*ell - 1) * x * P_lm1 - (ell + m - 1) * P_lm2) / (ell - m)
        P_lm2, P_lm1 = P_lm1, P_l

    return P_l


def d_m_legendre(l, m, x):
    """
    m-th derivative d^m/dx^m P_l(x) using associated Legendre.
    """
    if m < 0 or m > l:
        raise ValueError("Need 0 <= m <= l")
    Plm = P_lm(l, m, x)
    return (-1)**m * (1 - x**2)**(-m/2) * Plm


def f_legendre_1_1(x):
    return -1*np.sqrt(1-np.pow(x, 2))


def f_legendre_1_2(x):
    return -3*x*np.sqrt(1-np.pow(x, 2))


if __name__ == "__main__":
    x = np.linspace(0, 10, 1000)
    z = f_legendre_1_2(x) + 1
    y = P_lm(2, 1, x)
    plt.plot(x, y)
    plt.plot(x, z)
    plt.show()
