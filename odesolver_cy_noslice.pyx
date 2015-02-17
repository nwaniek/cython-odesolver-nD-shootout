cimport cython
import numpy as np
cimport numpy as np


cdef class Problem(object):
    cdef int N
    cdef double[:] dudt

    def __init__(self, int N):
        self.N = N
        self.dudt = np.zeros(N)

    cpdef double[:] rhs(self, double[:] u, double t):
        return np.zeros(self.N)


cdef class ExponentialDecayExample(Problem):
    def __init__(self):
        super(ExponentialDecayExample, self).__init__(5)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double[:] rhs(self, double[:] u, double t):
        self.dudt[0] = -10.0 * u[0]
        self.dudt[1] = -10.0 * u[1]
        self.dudt[2] = -10.0 * u[1]
        self.dudt[3] = -10.0 * u[1]
        self.dudt[4] = -10.0 * u[1]
        return self.dudt


cdef class Method(object):
    cdef int N
    cdef double[:] step_result
    cdef double[:] rhs

    def __init__(self, N):
        self.N = N
        self.step_result = np.zeros((N,), dtype=np.double)
        self.rhs = np.zeros((N,), dtype=np.double)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void step(self, double[:,::1] result, int i, int j, Problem f, double t, double dt):
        pass


cdef class MethodEuler(Method):
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void step(self, double[:,::1] result, int i, int j, Problem f, double t, double dt):
        self.rhs[:] = f.rhs(result[j, :], t)

        for n in range(self.N):
            result[i, n] = result[j, n] + dt * self.rhs[n]


cdef class MethodRK2(Method):
    cdef double[:] k1
    cdef double[:] tmp

    def __init__(self, N):
        super(MethodRK2, self).__init__(N)
        self.k1 = np.zeros((N,), dtype=np.double)
        self.tmp = np.zeros((N,), dtype=np.double)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void step(self, double[:,::1] result, int i, int j, Problem f, double t, double dt):
        self.rhs[:] = f.rhs(result[j, :], t)

        for n in range(self.N):
            self.k1[n] = dt * self.rhs[n]
            self.tmp[n] = result[j, n] + self.k1[n]

        self.rhs = f.rhs(self.tmp, t + dt)

        for n in range(self.N):
            result[i, n] = result[j, n] + 0.5 * (self.k1[n] + dt * self.rhs[n])


cdef class MethodRK4(Method):
    cdef double[:] k1
    cdef double[:] k2
    cdef double[:] k3
    cdef double[:] tmp

    def __init__(self, N):
        super(MethodRK4, self).__init__(N)
        self.k1 = np.zeros((N,), dtype=np.double)
        self.k2 = np.zeros((N,), dtype=np.double)
        self.k3 = np.zeros((N,), dtype=np.double)
        self.tmp = np.zeros((N,), dtype=np.double)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void step(self, double[:,::1] result, int i, int j, Problem f, double t, double dt):
        self.rhs[:] = f.rhs(result[j, :], t)

        # compute k1 and the next step
        for n in range(self.N):
            self.k1[n] = dt * self.rhs[n]
            self.tmp[n] = result[j, n] + 0.5 * self.k1[n]

        self.rhs[:] = f.rhs(self.tmp, t + 0.5 * dt)

        # compute k2 and the next step
        for n in range(self.N):
            self.k2[n] = dt * self.rhs[n]
            self.tmp[n] = result[j, n] + 0.5 * self.k2[n]

        self.rhs[:] = f.rhs(self.tmp, t + 0.5 * dt)

        # compute k3 and the next step
        for n in range(self.N):
            self.k3[n] = dt * self.rhs[n]
            self.tmp[n] = result[j, n] + self.k3[n]

        self.rhs[:] = f.rhs(self.tmp, t + dt)

        # compute k4 and combnne nt nnto the result
        for n in range(self.N):
            result[i, n] = result[j, n] + 1.0/6.0 * (
                    self.k1[n] + 2.0 * self.k2[n] + 2.0 * self.k3[n] + dt * self.rhs[n])


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double[:,:] __ODEsolve(Method method, Problem f, double a, double b, int N, double[::1] yInit):
    cdef double dt = (b - a) / N
    cdef double[:,::1] result = np.zeros((N+1, f.N), dtype=np.double)

    result[0, :] = yInit

    cdef double t
    cdef int i
    for i in range(1, N+1):
        t = a + i * dt
        method.step(result, i, i-1, f, t, dt)

    return result


cpdef double[:,:] ODEsolveRK4(Problem f, double a, double b, int N, double[::1] yInit):
    cdef Method method = MethodRK4(f.N)
    return __ODEsolve(method, f, a, b, N, yInit)


cpdef double[:,:] ODEsolveRK2(Problem f, double a, double b, int N, double[::1] yInit):
    cdef Method method = MethodRK2(f.N)
    return __ODEsolve(method, f, a, b, N, yInit)


cpdef double[:,:] ODEsolveEuler(Problem f, double a, double b, int N, double[::1] yInit):
    cdef Method method = MethodEuler(f.N)
    return __ODEsolve(method, f, a, b, N, yInit)


cpdef double[:,:] ODEsolve(Problem f, double a, double b, int N, double[::1] yInit):
    cdef Method method = MethodRK4(f.N)
    return __ODEsolve(method, f, a, b, N, yInit)


@cython.boundscheck(False)
cpdef double[:,:] test(int N):
    a = 0.0
    b = 1.0

    cdef int i
    cdef double[:,:] result
    cdef ExponentialDecayExample p = ExponentialDecayExample()
    cdef double[::1] yInit = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.double)

    result = ODEsolve(p, a, b, N, yInit)
    return result
