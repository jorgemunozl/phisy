"""
Vector field integration using Newton's second law.
F = m*dv/dt = m*a
So is very straight forward to integrate the velocity and position.

F_x = m*dv_x/dt
F_y = m*dv_y/dt

int F_xdt = m*dv_x
int F_ydt = m*dv_y

dv_x = int F_xdt/m
dv_y = int F_ydt/m

Wait integrate with respect to time is not the same as integrate with respect to position. Is hard!

So we are going to use the Euler method to integrate the velocity and position.
So we basically are doing the Finite Difference Method to integrate the velocity and position.

Now what else we can make? I think that we are going to use the Runge Kutta method and compare.

Because Runge Kutta is more accurate than Euler and more famous!.
"""
import numpy as np
import matplotlib.pyplot as plt


def force_field(x, y):
    """
    Example 2D force field F(x, y).

    Notes:
    - This is just an example; it is not necessarily conservative.
    - Units are arbitrary unless you assign them.
    """
    f1 = np.sin(x) + np.cos(y)
    f2 = 2*np.exp(-x**2 - y**2)
    # f3 = 3*x*y*z
    return np.array([f1, f2])



def plot_vector_field(X, Y, Z):
    plt.quiver(X, Y, Z[0], Z[1])


def simulate_particle_newton(force, *, mass, x0, v0, dt, t_end):
    """
    Forward Euler integration of:
      x' = v
      v' = F(x)/m
    """
    steps = int(np.ceil(t_end / dt))
    x = np.array(x0, dtype=float)
    v = np.array(v0, dtype=float)

    xs = np.empty((steps + 1, 2), dtype=float)
    vs = np.empty((steps + 1, 2), dtype=float)
    ts = np.linspace(0.0, steps * dt, steps + 1)
    xs[0] = x
    vs[0] = v

    for k in range(steps):
        F = force(x[0], x[1])
        a = F / mass
        v = v + dt * a
        x = x + dt * v
        xs[k + 1] = x
        vs[k + 1] = v

    return ts, xs, vs


def main():
    X = np.linspace(-2, 2, 15)
    Y = np.linspace(-2, 2, 15)
    X, Y = np.meshgrid(X, Y)
    Z = force_field(X, Y)

    mass = 1.0
    dt = 0.01
    t_end = 2.0
    initial_position = np.array([0.0, 1.0])
    initial_velocity = np.array([0, -2.0])

    ts, xs, _vs = simulate_particle_newton(
        force_field,
        mass=mass,
        x0=initial_position,
        v0=initial_velocity,
        dt=dt,
        t_end=t_end,
    )

    plt.figure()
    plot_vector_field(X, Y, Z)
    plt.plot(xs[:, 0], xs[:, 1], "-", linewidth=2, label="trajectory")
    plt.plot(xs[0, 0], xs[0, 1], "go", label="start")
    plt.plot(xs[-1, 0], xs[-1, 1], "ro", label="end")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Particle motion under F(x, y) with Newton's 2nd law")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()