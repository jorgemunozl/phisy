import matplotlib.pyplot as plt
import numpy as np


def diffraction(theta, wavelength, distance, N):
    arg = np.pi * distance / wavelength * np.sin(theta)
    return (np.sin(arg) / (arg)) ** 2


def intensity(theta, wavelength, distance, N):
    """
    Calculate the intensity of interference for a given angle, wavelength, distance, and number of waves.

    Parameters
    ----------
    theta : float
        The angle of incidence in radians.
    wavelength : float
        The wavelength of the light in meters.
    distance : float
        The distance between the two wave sources in meters.
    N : int
        The number of slits

    Returns
    -------
    float
        The intensity of the interference pattern.
    """
    arg = np.pi * distance / wavelength * np.sin(theta)
    return (np.sin(arg * N) / np.sin(arg)) ** 2


def main():
    theta = np.linspace(-np.pi / 6, np.pi / 6, 10000)
    wavelength = 5e-6  # 5 micrometers
    distance = 1e-3
    N = 10
    intensity_values = intensity(theta, wavelength, distance, N)
    plt.plot(theta, intensity_values)
    plt.xlabel("theta (radians)")
    plt.ylabel("intensity")
    plt.show()


if __name__ == "__main__":
    main()
