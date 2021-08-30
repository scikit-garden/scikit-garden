#! /usr/bin/env python

import os

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


DISTNAME = 'scikit-garden'
DESCRIPTION = "A garden of scikit-learn compatible trees"
URL = 'https://github.com/scikit-garden/scikit-garden'
MAINTAINER = 'Manoj Kumar'
MAINTAINER_EMAIL = 'mks542@nyu.edu'
LICENSE = 'new BSD'
VERSION = '0.1.3'

libraries = []
if os.name == 'posix':
    libraries.append('m')

extensions = []
for name in ['_tree', '_splitter', '_criterion', '_utils']:
    extensions.append(Extension(
        'skgarden.mondrian.tree.{}'.format(name),
        sources=['skgarden/mondrian/tree/{}.pyx'.format(name)],
        libraries=libraries,
        extra_compile_args=['-O3'],
    ))


class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):
        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        super().run()

    def finalize_options(self):
        # Import Cython here, only when we need to cythonize extensions
        if self.distribution.ext_modules:
            from Cython.Build.Dependencies import cythonize
            self.distribution.ext_modules[:] = cythonize(self.distribution.ext_modules, force=self.force)
        super().finalize_options()


if __name__ == "__main__":
    setup(name=DISTNAME,
          cmdclass={'build_ext': CustomBuildExtCommand},
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          packages=find_packages(),
          include_package_data=True,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'
            ],
          install_requires=["numpy", "scipy", "scikit-learn>=0.22", "six"],
          setup_requires=["Cython>=0.23", "numpy", "setuptools>=18"],
          ext_modules=extensions,
          )
