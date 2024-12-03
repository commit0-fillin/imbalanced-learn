"""This is a copy of sklearn/utils/tests/test_param_validation.py. It can be
removed when we support scikit-learn >= 1.2.
"""
from numbers import Integral, Real
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import deprecated
from imblearn._config import config_context, get_config
from imblearn.base import _ParamsValidationMixin
from imblearn.utils._param_validation import HasMethods, Hidden, Interval, InvalidParameterError, MissingValues, Options, RealNotInt, StrOptions, _ArrayLikes, _Booleans, _Callables, _CVObjects, _InstancesOf, _IterablesNotString, _NanConstraint, _NoneConstraint, _PandasNAConstraint, _RandomStates, _SparseMatrices, _VerboseHelper, generate_invalid_param_val, generate_valid_param, make_constraint, validate_params
from imblearn.utils.fixes import _fit_context

@validate_params({'a': [Real], 'b': [Real], 'c': [Real], 'd': [Real]}, prefer_skip_nested_validation=True)
def _func(a, b=0, *args, c, d=0, **kwargs):
    """A function to test the validation of functions."""
    return a + b + c + d

class _Class:
    """A class to test the _InstancesOf constraint and the validation of methods."""

    @validate_params({'a': [Real]}, prefer_skip_nested_validation=True)
    def _method(self, a):
        """A validated method"""
        return a * 2

    @deprecated()
    @validate_params({'a': [Real]}, prefer_skip_nested_validation=True)
    def _deprecated_method(self, a):
        """A deprecated validated method"""
        return a + 1

class _Estimator(_ParamsValidationMixin, BaseEstimator):
    """An estimator to test the validation of estimator parameters."""
    _parameter_constraints: dict = {'a': [Real]}

    def __init__(self, a):
        self.a = a

@pytest.mark.parametrize('interval_type', [Integral, Real])
def test_interval_range(interval_type):
    """Check the range of values depending on closed."""
    interval = Interval(interval_type, 0, 5, closed='both')
    assert 0 in interval
    assert 5 in interval
    
    interval = Interval(interval_type, 0, 5, closed='left')
    assert 0 in interval
    assert 5 not in interval
    
    interval = Interval(interval_type, 0, 5, closed='right')
    assert 0 not in interval
    assert 5 in interval
    
    interval = Interval(interval_type, 0, 5, closed='neither')
    assert 0 not in interval
    assert 5 not in interval

@pytest.mark.parametrize('interval_type', [Integral, Real])
def test_interval_large_integers(interval_type):
    """Check that Interval constraint work with large integers.

    non-regression test for #26648.
    """
    large_int = 2**63 - 1
    interval = Interval(interval_type, -large_int, large_int, closed='both')
    assert -large_int in interval
    assert large_int in interval
    assert large_int + 1 not in interval

def test_interval_inf_in_bounds():
    """Check that inf is included iff a bound is closed and set to None.

    Only valid for real intervals.
    """
    interval = Interval(Real, None, None, closed='both')
    assert np.inf in interval
    assert -np.inf in interval

    interval = Interval(Real, None, 0, closed='left')
    assert -np.inf in interval
    assert np.inf not in interval

    interval = Interval(Real, 0, None, closed='right')
    assert np.inf in interval
    assert -np.inf not in interval

    interval = Interval(Real, None, None, closed='neither')
    assert np.inf not in interval
    assert -np.inf not in interval

@pytest.mark.parametrize('interval', [Interval(Real, 0, 1, closed='left'), Interval(Real, None, None, closed='both')])
def test_nan_not_in_interval(interval):
    """Check that np.nan is not in any interval."""
    assert np.nan not in interval

@pytest.mark.parametrize('params, error, match', [({'type': Integral, 'left': 1.0, 'right': 2, 'closed': 'both'}, TypeError, 'Expecting left to be an int for an interval over the integers'), ({'type': Integral, 'left': 1, 'right': 2.0, 'closed': 'neither'}, TypeError, 'Expecting right to be an int for an interval over the integers'), ({'type': Integral, 'left': None, 'right': 0, 'closed': 'left'}, ValueError, "left can't be None when closed == left"), ({'type': Integral, 'left': 0, 'right': None, 'closed': 'right'}, ValueError, "right can't be None when closed == right"), ({'type': Integral, 'left': 1, 'right': -1, 'closed': 'both'}, ValueError, "right can't be less than left")])
def test_interval_errors(params, error, match):
    """Check that informative errors are raised for invalid combination of parameters"""
    with pytest.raises(error, match=match):
        Interval(**params)

def test_stroptions():
    """Sanity check for the StrOptions constraint"""
    options = StrOptions({'a', 'b', 'c'})
    assert 'a' in options.options
    assert 'd' not in options.options
    assert str(options) == "a str among {'a', 'b', 'c'}"

    options_with_deprecated = StrOptions({'a', 'b', 'c'}, deprecated={'b'})
    assert str(options_with_deprecated) == "a str among {'a', 'b (deprecated)', 'c'}"

def test_options():
    """Sanity check for the Options constraint"""
    options = Options(int, {1, 2, 3})
    assert 1 in options.options
    assert 4 not in options.options
    assert str(options) == "an int among {1, 2, 3}"

    options_with_deprecated = Options(int, {1, 2, 3}, deprecated={2})
    assert str(options_with_deprecated) == "an int among {1, 2 (deprecated), 3}"

@pytest.mark.parametrize('type, expected_type_name', [(int, 'int'), (Integral, 'int'), (Real, 'float'), (np.ndarray, 'numpy.ndarray')])
def test_instances_of_type_human_readable(type, expected_type_name):
    """Check the string representation of the _InstancesOf constraint."""
    constraint = _InstancesOf(type)
    assert str(constraint) == f"an instance of '{expected_type_name}'"

def test_hasmethods():
    """Check the HasMethods constraint."""
    constraint = HasMethods(['fit', 'predict'])
    
    class ValidEstimator:
        def fit(self): pass
        def predict(self): pass
    
    class InvalidEstimator:
        def fit(self): pass
    
    assert constraint.is_satisfied_by(ValidEstimator())
    assert not constraint.is_satisfied_by(InvalidEstimator())
    assert str(constraint) == "an object implementing 'fit' and 'predict'"

@pytest.mark.parametrize('constraint', [Interval(Real, None, 0, closed='left'), Interval(Real, 0, None, closed='left'), Interval(Real, None, None, closed='neither'), StrOptions({'a', 'b', 'c'}), MissingValues(), MissingValues(numeric_only=True), _VerboseHelper(), HasMethods('fit'), _IterablesNotString(), _CVObjects()])
def test_generate_invalid_param_val(constraint):
    """Check that the value generated does not satisfy the constraint"""
    pass

@pytest.mark.parametrize('integer_interval, real_interval', [(Interval(Integral, None, 3, closed='right'), Interval(RealNotInt, -5, 5, closed='both')), (Interval(Integral, None, 3, closed='right'), Interval(RealNotInt, -5, 5, closed='neither')), (Interval(Integral, None, 3, closed='right'), Interval(RealNotInt, 4, 5, closed='both')), (Interval(Integral, None, 3, closed='right'), Interval(RealNotInt, 5, None, closed='left')), (Interval(Integral, None, 3, closed='right'), Interval(RealNotInt, 4, None, closed='neither')), (Interval(Integral, 3, None, closed='left'), Interval(RealNotInt, -5, 5, closed='both')), (Interval(Integral, 3, None, closed='left'), Interval(RealNotInt, -5, 5, closed='neither')), (Interval(Integral, 3, None, closed='left'), Interval(RealNotInt, 1, 2, closed='both')), (Interval(Integral, 3, None, closed='left'), Interval(RealNotInt, None, -5, closed='left')), (Interval(Integral, 3, None, closed='left'), Interval(RealNotInt, None, -4, closed='neither')), (Interval(Integral, -5, 5, closed='both'), Interval(RealNotInt, None, 1, closed='right')), (Interval(Integral, -5, 5, closed='both'), Interval(RealNotInt, 1, None, closed='left')), (Interval(Integral, -5, 5, closed='both'), Interval(RealNotInt, -10, -4, closed='neither')), (Interval(Integral, -5, 5, closed='both'), Interval(RealNotInt, -10, -4, closed='right')), (Interval(Integral, -5, 5, closed='neither'), Interval(RealNotInt, 6, 10, closed='neither')), (Interval(Integral, -5, 5, closed='neither'), Interval(RealNotInt, 6, 10, closed='left')), (Interval(Integral, 2, None, closed='left'), Interval(RealNotInt, 0, 1, closed='both')), (Interval(Integral, 1, None, closed='left'), Interval(RealNotInt, 0, 1, closed='both'))])
def test_generate_invalid_param_val_2_intervals(integer_interval, real_interval):
    """Check that the value generated for an interval constraint does not satisfy any of
    the interval constraints.
    """
    pass

@pytest.mark.parametrize('constraint', [_ArrayLikes(), _InstancesOf(list), _Callables(), _NoneConstraint(), _RandomStates(), _SparseMatrices(), _Booleans(), Interval(Integral, None, None, closed='neither')])
def test_generate_invalid_param_val_all_valid(constraint):
    """Check that the function raises NotImplementedError when there's no invalid value
    for the constraint.
    """
    pass

@pytest.mark.parametrize('constraint', [_ArrayLikes(), _Callables(), _InstancesOf(list), _NoneConstraint(), _RandomStates(), _SparseMatrices(), _Booleans(), _VerboseHelper(), MissingValues(), MissingValues(numeric_only=True), StrOptions({'a', 'b', 'c'}), Options(Integral, {1, 2, 3}), Interval(Integral, None, None, closed='neither'), Interval(Integral, 0, 10, closed='neither'), Interval(Integral, 0, None, closed='neither'), Interval(Integral, None, 0, closed='neither'), Interval(Real, 0, 1, closed='neither'), Interval(Real, 0, None, closed='both'), Interval(Real, None, 0, closed='right'), HasMethods('fit'), _IterablesNotString(), _CVObjects()])
def test_generate_valid_param(constraint):
    """Check that the value generated does satisfy the constraint."""
    pass

@pytest.mark.parametrize('constraint_declaration, value', [(Interval(Real, 0, 1, closed='both'), 0.42), (Interval(Integral, 0, None, closed='neither'), 42), (StrOptions({'a', 'b', 'c'}), 'b'), (Options(type, {np.float32, np.float64}), np.float64), (callable, lambda x: x + 1), (None, None), ('array-like', [[1, 2], [3, 4]]), ('array-like', np.array([[1, 2], [3, 4]])), ('sparse matrix', csr_matrix([[1, 2], [3, 4]])), ('random_state', 0), ('random_state', np.random.RandomState(0)), ('random_state', None), (_Class, _Class()), (int, 1), (Real, 0.5), ('boolean', False), ('verbose', 1), ('nan', np.nan), (MissingValues(), -1), (MissingValues(), -1.0), (MissingValues(), 2 ** 1028), (MissingValues(), None), (MissingValues(), float('nan')), (MissingValues(), np.nan), (MissingValues(), 'missing'), (HasMethods('fit'), _Estimator(a=0)), ('cv_object', 5)])
def test_is_satisfied_by(constraint_declaration, value):
    """Sanity check for the is_satisfied_by method"""
    pass

@pytest.mark.parametrize('constraint_declaration, expected_constraint_class', [(Interval(Real, 0, 1, closed='both'), Interval), (StrOptions({'option1', 'option2'}), StrOptions), (Options(Real, {0.42, 1.23}), Options), ('array-like', _ArrayLikes), ('sparse matrix', _SparseMatrices), ('random_state', _RandomStates), (None, _NoneConstraint), (callable, _Callables), (int, _InstancesOf), ('boolean', _Booleans), ('verbose', _VerboseHelper), (MissingValues(numeric_only=True), MissingValues), (HasMethods('fit'), HasMethods), ('cv_object', _CVObjects), ('nan', _NanConstraint)])
def test_make_constraint(constraint_declaration, expected_constraint_class):
    """Check that make_constraint dispatches to the appropriate constraint class"""
    pass

def test_make_constraint_unknown():
    """Check that an informative error is raised when an unknown constraint is passed"""
    pass

def test_validate_params():
    """Check that validate_params works no matter how the arguments are passed"""
    pass

def test_validate_params_missing_params():
    """Check that no error is raised when there are parameters without
    constraints
    """
    pass

def test_decorate_validated_function():
    """Check that validate_params functions can be decorated"""
    pass

def test_validate_params_method():
    """Check that validate_params works with methods"""
    pass

def test_validate_params_estimator():
    """Check that validate_params works with Estimator instances"""
    pass

def test_stroptions_deprecated_subset():
    """Check that the deprecated parameter must be a subset of options."""
    pass

def test_hidden_constraint():
    """Check that internal constraints are not exposed in the error message."""
    pass

def test_hidden_stroptions():
    """Check that we can have 2 StrOptions constraints, one being hidden."""
    pass

def test_validate_params_set_param_constraints_attribute():
    """Check that the validate_params decorator properly sets the parameter constraints
    as attribute of the decorated function/method.
    """
    pass

def test_boolean_constraint_deprecated_int():
    """Check that validate_params raise a deprecation message but still passes
    validation when using an int for a parameter accepting a boolean.
    """
    pass

def test_no_validation():
    """Check that validation can be skipped for a parameter."""
    pass

def test_pandas_na_constraint_with_pd_na():
    """Add a specific test for checking support for `pandas.NA`."""
    pass

def test_iterable_not_string():
    """Check that a string does not satisfy the _IterableNotString constraint."""
    pass

def test_cv_objects():
    """Check that the _CVObjects constraint accepts all current ways
    to pass cv objects."""
    pass

def test_third_party_estimator():
    """Check that the validation from a scikit-learn estimator inherited by a third
    party estimator does not impose a match between the dict of constraints and the
    parameters of the estimator.
    """
    pass

def test_interval_real_not_int():
    """Check for the type RealNotInt in the Interval constraint."""
    pass

def test_real_not_int():
    """Check for the RealNotInt type."""
    pass

def test_skip_param_validation():
    """Check that param validation can be skipped using config_context."""
    pass

@pytest.mark.parametrize('prefer_skip_nested_validation', [True, False])
def test_skip_nested_validation(prefer_skip_nested_validation):
    """Check that nested validation can be skipped."""
    pass

@pytest.mark.parametrize('skip_parameter_validation, prefer_skip_nested_validation, expected_skipped', [(True, True, True), (True, False, True), (False, True, True), (False, False, False)])
def test_skip_nested_validation_and_config_context(skip_parameter_validation, prefer_skip_nested_validation, expected_skipped):
    """Check interaction between global skip and local skip."""
    pass
