"""Utilities for deprecation"""
import warnings

def deprecate_parameter(sampler, version_deprecation, param_deprecated, new_param=None):
    """Helper to deprecate a parameter by another one.

    Parameters
    ----------
    sampler : sampler object,
        The object which will be inspected.

    version_deprecation : str,
        The version from which the parameter will be deprecated. The format
        should be ``'x.y'``.

    param_deprecated : str,
        The parameter being deprecated.

    new_param : str,
        The parameter used instead of the deprecated parameter. By default, no
        parameter is expected.
    """
    if hasattr(sampler, param_deprecated):
        if new_param is None:
            message = (
                f"The parameter '{param_deprecated}' is deprecated in "
                f"version {version_deprecation} and will be removed in a "
                f"future version."
            )
        else:
            message = (
                f"The parameter '{param_deprecated}' is deprecated in "
                f"version {version_deprecation} and will be removed in a "
                f"future version. Use '{new_param}' instead."
            )
        warnings.warn(message, category=FutureWarning)
