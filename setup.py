from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("toy_model_mc.pyx")
)