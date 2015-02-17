"""Generic setup.py to compile each .pyx module"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

extensions = cythonize([
    Extension(
        "*",
        sources=["*.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args = ["-ffast-math", "-O3"],
        libraries=["m"],
    ),
])

setup(
    ext_modules = extensions,
)
