import numpy as np


def fn(x, t):
    return -10.0 * x


# Euler forward algorithm
def ODEstepEuler(yn, f, t, dt):
    return yn + dt * f(yn, t)


# 2nd order Runge Kutta step
def ODEstepRK2(yn, f, t, dt):
    k1 = dt * f(yn, t)
    k2 = dt * f(yn + k1, t + dt)
    return yn + 0.5 * (k1 + k2)


# 4th order Runge Kutta step
def ODEstepRK4(yn, f, t, dt):
    k1 = dt * f(yn, t)
    k2 = dt * f(yn + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * f(yn + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * f(yn + k3, t + dt)
    return yn + 1.0/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)


def ODEsolve(f, a, b, N, yInit):
    dt = (b - a) / N
    result = np.zeros((N+1, 5))
    result[0, :] = yInit
    for i in range(N+1):
        t = a + i * dt
        result[i, :] = ODEstepRK4(result[i-1, :], f, t, dt)
    return result


def test(N):
    a = 0.0
    b = 1.0
    yInit = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.double)
    result = ODEsolve(fn, a, b, N, yInit)
    return result
