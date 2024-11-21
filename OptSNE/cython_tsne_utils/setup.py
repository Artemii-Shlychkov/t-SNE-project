from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

extra_compile_args = [
    "-O3",
    "-ffast-math",
    "-fopenmp",
    "-march=native",
    "-mtune=native",
    "-floop-parallelize-all",
    "-funroll-loops",
]
extra_link_args = ["-fopenmp"]

os.environ["CC"] = "gcc-14"
os.environ["CXX"] = "g++-14"

extensions = [
    Extension(
        name="quad_tree",
        sources=["quad_tree.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
    Extension(
        name="tsne_bh",
        sources=["tsne_bh.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    ),
]

setup(
    name="tsne_bh_package",
    ext_modules=cythonize(extensions, language_level=3),
    zip_safe=False,
)
