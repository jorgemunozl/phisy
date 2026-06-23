"""
Riemann sum for a line integral
"""
import numpy as np
import matplotlib.pyplot as plt

def function(x: float):
    return np.log(np.exp(x) + np.sin(x)**2)*np.sqrt(np.exp(2*x) + 4*np.cos(x)**2)

N = 1000
start = 0
end = 1
x = np.linspace(start, end, N)
y = function(x)

integral = np.cumsum(y) * (x[1] - x[0])

dxdy = np.gradient(y, x)

def integrate(function, start, end, N):
    delta_x = (end - start) / N
    x_i = np.linspace(start, end, N)
    return np.sum(function(x_i) * delta_x)

integral_riemann = integrate(function, start, end, N)

manual_integral = []
values = np.linspace(start, end, N)
for i in values:
    integral_i = integrate(function, start, i, N)
    manual_integral.append(integral_i)

manual_integral = np.array(manual_integral)


print("manual_integral: ", manual_integral[-1])
print("trapezoidal integral: ", np.trapezoid(y, x))
plt.plot(values, manual_integral, label="manual integral")
plt.plot(x, y, label="function")
plt.plot(x, integral, label="integral")
plt.plot(x, dxdy, label="derivative")
plt.legend()
plt.show()