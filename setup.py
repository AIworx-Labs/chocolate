import sys

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError


if sys.platform == 'win32' and sys.version_info > (2, 6):
    # 2.6's distutils.msvc9compiler can raise an IOError when failing to
    # find the compiler
    # It can also raise ValueError http://bugs.python.org/issue7511
    ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError, IOError, ValueError)
else:
    ext_errors = (CCompilerError, DistutilsExecError, DistutilsPlatformError)


class BuildFailed(Exception):
    pass


class ve_build_ext(build_ext):
    # This class allows C extension building to fail.

    def run(self):
        try:
            build_ext.run(self)
        except DistutilsPlatformError:
            raise BuildFailed()

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except ext_errors:
            raise BuildFailed()

def run_setup(build_ext):
    extra_modules = None
    if build_ext:
        extra_modules = list()

        hv_module = Extension("chocolate.mo._hv", sources=["chocolate/mo/_hv.c", "chocolate/mo/hv.cpp"])
        extra_modules.append(hv_module)

    setup(
        name="chocolate",
        version="0.6",
        packages=find_packages(exclude=['examples', 'tests']),
        test_suite="tests",
        install_requires=["numpy>=1.11", "scipy>=0.18", "scikit-learn>=0.18", "pandas>=0.19", "dataset>=0.8", "filelock>=2.0"],
        author="François-Michel De Rainville, Olivier Gagnon",
        author_email="chocolate@novasyst.com",
        description="Asynchrone hyperparameter optimization",
        license="BSD 3-clauses",
        keywords="AsynchroneHyperparameter Optimizer",
        url="http://github.com/NovaSyst/chocolate",
        ext_modules=extra_modules,
        cmdclass={"build_ext" : ve_build_ext}
    )

try:
    run_setup(True)
except BuildFailed:
    run_setup(False)

    print("*" * 75)
    print("WARNING: The hypervolume C extension could not be compiled, speedups won't be available.")
    print("Plain-Python installation succeeded.")
    print("*" * 75)