#!/usr/bin/env python

import os, sys, timeit, platform

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


def shootout():

    setup = '''
from math import sin
from odesolver_%s import test
'''

    stmt = '''
test(100)
'''

    repeat = 10
    number = 1000

    methods = ['pure python', 'pure cythonized', 'sliced cython', 'no-slice cython', 'pointer cython']
    imports = ['py', 'cy_pure', 'cy', 'cy_noslice', 'cy_ptr']

    # time all variants
    ts = np.zeros(len(imports))
    for i, s in enumerate(imports):
        print("running method '%s'" % methods[i])
        ts[i] = min(timeit.repeat(stmt, setup % s, repeat=repeat, number=number))

    # get speed comparisons
    ds = ts[0] / ts

    # emit hardware information and statistics
    print("platform: ", platform.processor())
    print("method\t\t\truntime\tspeedup")
    print("----------------------------------------")
    for i, m in enumerate(methods):
        print("%s\t\t%6.4f\t%6.4f" % (m, ts[i], ds[i]))


    # show nice little figure
    ind = np.arange(len(methods))
    width = 0.65
    fig = plt.figure()
    ax = fig.add_subplot(111)

    fig.suptitle("Comparison of ODE solver optimizations", fontsize=13, fontweight="bold")
    ax.set_title(platform.processor())
    ax.set_xlabel('Implementation', fontweight="bold")
    ax.set_ylabel('Speed Up', fontweight="bold")
    bars = ax.bar(ind, ds, width, color="#00719a")

    i = 0
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, 1.02 * h, '%4.2f' % ds[i], ha='center', va='bottom')
        i += 1

    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(methods, rotation=70, ha='center')
    ax.axhline(1.0, color='#8a8a8a')
    fig.subplots_adjust(bottom=0.25)

    plt.show()

if __name__ == "__main__":
    shootout()
