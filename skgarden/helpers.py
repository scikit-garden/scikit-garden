## This is copied from sklearn version 0.20 as this has been removed in later sklearn versions.

def _joblib_parallel_args(**kwargs):
        """
        Copied from an old sklearn version
        
        Set joblib.Parallel arguments in a compatible way for 0.11 and 0.12+
        For joblib 0.11 this maps both ``prefer`` and ``require`` parameters to
        a specific ``backend``.
        Parameters
        ----------
        prefer : str in {'processes', 'threads'} or None
            Soft hint to choose the default backend if no specific backend
            was selected with the parallel_backend context manager.
        require : 'sharedmem' or None
            Hard condstraint to select the backend. If set to 'sharedmem',
            the selected backend will be single-host and thread-based even
            if the user asked for a non-thread based backend with
            parallel_backend.
        See joblib.Parallel documentation for more details
        """
        from . import _joblib

        if _joblib.__version__ >= LooseVersion('0.12'):
            return kwargs

        extra_args = set(kwargs.keys()).difference({'prefer', 'require'})
        if extra_args:
            raise NotImplementedError('unhandled arguments %s with joblib %s'
                                  % (list(extra_args), _joblib.__version__))
        args = {}
        if 'prefer' in kwargs:
            prefer = kwargs['prefer']
            if prefer not in ['threads', 'processes', None]:
                raise ValueError('prefer=%s is not supported' % prefer)
            args['backend'] = {'threads': 'threading',
                               'processes': 'multiprocessing',
                               None: None}[prefer]

        if 'require' in kwargs:
            require = kwargs['require']
            if require not in [None, 'sharedmem']:
                raise ValueError('require=%s is not supported' % require)
            if require == 'sharedmem':
                args['backend'] = 'threading'
        return args
    