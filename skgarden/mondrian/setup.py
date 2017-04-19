from distutils.version import LooseVersion

CYTHON_MIN_VERSION = '0.23'

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('mondrian', parent_package, top_path)
    config.add_subpackage('tree')
    config.add_subpackage('ensemble')

    message = ('Please install cython with a version >= {0} in order '
               'to build a scikit-learn development version.').format(
               CYTHON_MIN_VERSION)
    try:
        import Cython
        if LooseVersion(Cython.__version__) < CYTHON_MIN_VERSION:
            message += ' Your version of Cython was {0}.'.format(
                Cython.__version__)
            raise ValueError(message)
        from Cython.Build import cythonize
    except ImportError as exc:
        exc.args += (message,)
        raise
    config.ext_modules = cythonize(config.ext_modules)

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
