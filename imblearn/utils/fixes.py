"""Compatibility fixes for older version of python, numpy, scipy, and
scikit-learn.

If you add content to this file, please give the version of the package at
which the fix is no longer needed.
"""
import functools
import sys
import numpy as np
import scipy
import scipy.stats
import sklearn
from sklearn.utils.fixes import parse_version
from .._config import config_context, get_config
sp_version = parse_version(scipy.__version__)
sklearn_version = parse_version(sklearn.__version__)
if sklearn_version >= parse_version('1.1'):
    from sklearn.utils.validation import _is_arraylike_not_scalar
else:
    from sklearn.utils.validation import _is_arraylike

    def _is_arraylike_not_scalar(array):
        """Return True if array is array-like and not a scalar"""
        return _is_arraylike(array) and not np.isscalar(array)
if sklearn_version < parse_version('1.3'):

    def _fit_context(*, prefer_skip_nested_validation):
        """Decorator to run the fit methods of estimators within context managers.

        Parameters
        ----------
        prefer_skip_nested_validation : bool
            If True, the validation of parameters of inner estimators or functions
            called during fit will be skipped.

            This is useful to avoid validating many times the parameters passed by the
            user from the public facing API. It's also useful to avoid validating
            parameters that we pass internally to inner functions that are guaranteed to
            be valid by the test suite.

            It should be set to True for most estimators, except for those that receive
            non-validated objects as parameters, such as meta-estimators that are given
            estimator objects.

        Returns
        -------
        decorated_fit : method
            The decorated fit method.
        """
        def decorator(fit_method):
            @functools.wraps(fit_method)
            def wrapper(self, *args, **kwargs):
                with config_context(skip_parameter_validation=prefer_skip_nested_validation):
                    return fit_method(self, *args, **kwargs)
            return wrapper
        return decorator
else:
    from sklearn.base import _fit_context
if sklearn_version < parse_version('1.3'):

    def _is_fitted(estimator, attributes=None, all_or_any=all):
        """Determine if an estimator is fitted

        Parameters
        ----------
        estimator : estimator instance
            Estimator instance for which the check is performed.

        attributes : str, list or tuple of str, default=None
            Attribute name(s) given as string or a list/tuple of strings
            Eg.: ``["coef_", "estimator_", ...], "coef_"``

            If `None`, `estimator` is considered fitted if there exist an
            attribute that ends with a underscore and does not start with double
            underscore.

        all_or_any : callable, {all, any}, default=all
            Specify whether all or any of the given attributes must exist.

        Returns
        -------
        fitted : bool
            Whether the estimator is fitted.
        """
        if attributes is None:
            attributes = [attr for attr in vars(estimator)
                          if attr.endswith("_") and not attr.startswith("__")]
        
        if isinstance(attributes, str):
            attributes = [attributes]
        
        return all_or_any(hasattr(estimator, attr) for attr in attributes)
else:
    from sklearn.utils.validation import _is_fitted
try:
    from sklearn.utils.validation import _is_pandas_df
except ImportError:

    def _is_pandas_df(X):
        """Return True if the X is a pandas dataframe."""
        try:
            import pandas as pd
            return isinstance(X, pd.DataFrame)
        except ImportError:
            return False
