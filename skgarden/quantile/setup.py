import os

import numpy
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("quantile", parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('m')
    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path="").todict())
