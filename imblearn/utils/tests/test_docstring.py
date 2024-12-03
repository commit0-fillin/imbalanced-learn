"""Test utilities for docstring."""
import sys
import textwrap
import pytest
from imblearn.utils import Substitution
from imblearn.utils._docstring import _n_jobs_docstring, _random_state_docstring

def _dedent_docstring(docstring):
    """Compatibility with Python 3.13+.

    xref: https://github.com/python/cpython/issues/81283
    """
    return textwrap.dedent(docstring)
func_docstring = 'A function.\n\n    Parameters\n    ----------\n    xxx\n\n    yyy\n    '

def func(param_1, param_2):
    """A function.

    Parameters
    ----------
    {param_1}

    {param_2}
    """
    return param_1, param_2
cls_docstring = 'A class.\n\n    Parameters\n    ----------\n    xxx\n\n    yyy\n    '

class cls:
    """A class.

    Parameters
    ----------
    {param_1}

    {param_2}
    """

    def __init__(self, param_1, param_2):
        self.param_1 = param_1
        self.param_2 = param_2
if sys.version_info >= (3, 13):
    func_docstring = _dedent_docstring(func_docstring)
    cls_docstring = _dedent_docstring(cls_docstring)

def test_docstring_with_python_OO():
    """Check that we don't raise a warning if the code is executed with -OO.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/issues/945
    """
    # Test that docstrings are not stripped when running with -OO
    assert func.__doc__ is not None
    assert cls.__doc__ is not None
    
    # Test that Substitution still works with -OO
    sub = Substitution(param_1="Parameter 1", param_2="Parameter 2")
    decorated_func = sub(func)
    decorated_cls = sub(cls)
    
    assert "Parameter 1" in decorated_func.__doc__
    assert "Parameter 2" in decorated_func.__doc__
    assert "Parameter 1" in decorated_cls.__doc__
    assert "Parameter 2" in decorated_cls.__doc__
