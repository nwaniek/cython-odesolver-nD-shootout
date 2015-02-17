#!/usr/bin/env python

import os, sys

# make sure the modules are compiled# {{{
try:
    print("building cython modules if necessary")
    assert(os.system("python setup.py build_ext -i") == 0)
except:
    print("unable to build the cython modules")
    sys.exit(1)
# }}}

import numpy as np
import matplotlib.pyplot as plt

from odesolver_cy_ptr import Problem, ODEsolve

class MyProblem(Problem):
    def __init__(self):
        super(MyProblem, self).__init__(10)

    def rhs(self, u, t):
        return -10.0 * np.asarray(u)

def main():
    p = MyProblem()
    a = 0.0
    b = 1.0
    N = 100
    M = 10
    yInit = np.linspace(0, 1, M)
    result = ODEsolve(p, a, b, N, yInit)

    plt.figure()
    for i in range(M):
        plt.plot(np.linspace(0, 1, N+1), result[:, i])
    plt.show()


if __name__ == "__main__":
    main()
