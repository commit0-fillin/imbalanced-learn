"""This is a copy of sklearn/utils/_param_validation.py. It can be removed when
we support scikit-learn >= 1.2.
"""
import functools
import math
import operator
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from inspect import signature
from numbers import Integral, Real
import numpy as np
import sklearn
from scipy.sparse import csr_matrix, issparse
from sklearn.utils.fixes import parse_version
from .._config import config_context, get_config
from ..utils.fixes import _is_arraylike_not_scalar
sklearn_version = parse_version(sklearn.__version__)
if sklearn_version < parse_version('1.4'):

    class InvalidParameterError(ValueError, TypeError):
        """Custom exception to be raised when the parameter of a class/method/function
        does not have a valid type or value.
        """

    def validate_parameter_constraints(parameter_constraints, params, caller_name):
        """Validate types and values of given parameters.

        Parameters
        ----------
        parameter_constraints : dict or {"no_validation"}
            If "no_validation", validation is skipped for this parameter.

            If a dict, it must be a dictionary `param_name: list of constraints`.
            A parameter is valid if it satisfies one of the constraints from the list.
            Constraints can be:
            - an Interval object, representing a continuous or discrete range of numbers
            - the string "array-like"
            - the string "sparse matrix"
            - the string "random_state"
            - callable
            - None, meaning that None is a valid value for the parameter
            - any type, meaning that any instance of this type is valid
            - an Options object, representing a set of elements of a given type
            - a StrOptions object, representing a set of strings
            - the string "boolean"
            - the string "verbose"
            - the string "cv_object"
            - the string "nan"
            - a MissingValues object representing markers for missing values
            - a HasMethods object, representing method(s) an object must have
            - a Hidden object, representing a constraint not meant to be exposed to the
              user

        params : dict
            A dictionary `param_name: param_value`. The parameters to validate against
            the constraints.

        caller_name : str
            The name of the estimator or function or method that called this function.
        """
        if parameter_constraints == "no_validation":
            return

        for param_name, param_value in params.items():
            if param_name not in parameter_constraints:
                raise ValueError(f"Unknown parameter {param_name} for {caller_name}")

            constraints = parameter_constraints[param_name]
            if not isinstance(constraints, list):
                constraints = [constraints]

            for constraint in constraints:
                if isinstance(constraint, Interval):
                    if constraint.type == Integral:
                        if not isinstance(param_value, Integral):
                            continue
                    elif constraint.type == Real:
                        if not isinstance(param_value, Real):
                            continue
                    if param_value in constraint:
                        break
                elif constraint == "array-like":
                    if _is_arraylike_not_scalar(param_value):
                        break
                elif constraint == "sparse matrix":
                    if issparse(param_value):
                        break
                elif constraint == "random_state":
                    if isinstance(param_value, (int, np.random.RandomState, type(None))):
                        break
                elif callable(constraint):
                    if callable(param_value):
                        break
                elif constraint is None:
                    if param_value is None:
                        break
                elif isinstance(constraint, type):
                    if isinstance(param_value, constraint):
                        break
                elif isinstance(constraint, Options):
                    if isinstance(param_value, constraint.type) and param_value in constraint.options:
                        break
                elif isinstance(constraint, StrOptions):
                    if isinstance(param_value, str) and param_value in constraint.options:
                        break
                elif constraint == "boolean":
                    if isinstance(param_value, bool):
                        break
                elif constraint == "verbose":
                    if isinstance(param_value, (bool, int)):
                        break
                elif constraint == "cv_object":
                    if isinstance(param_value, (int, HasMethods(["split", "get_n_splits"]), _IterablesNotString, type(None))):
                        break
                elif constraint == "nan":
                    if np.isnan(param_value):
                        break
                elif isinstance(constraint, MissingValues):
                    if param_value in constraint._constraints:
                        break
                elif isinstance(constraint, HasMethods):
                    if all(hasattr(param_value, method) for method in constraint.methods):
                        break
                elif isinstance(constraint, Hidden):
                    # Hidden constraints are not validated
                    break
            else:
                raise InvalidParameterError(
                    f"The {param_name} parameter of {caller_name} must be "
                    f"{' or '.join(str(c) for c in constraints)}. Got {param_value!r} instead."
                )

    def make_constraint(constraint):
        """Convert the constraint into the appropriate Constraint object.

        Parameters
        ----------
        constraint : object
            The constraint to convert.

        Returns
        -------
        constraint : instance of _Constraint
            The converted constraint.
        """
        if isinstance(constraint, _Constraint):
            return constraint
        elif constraint == "array-like":
            return _ArrayLikes()
        elif constraint == "sparse matrix":
            return _SparseMatrices()
        elif constraint == "random_state":
            return _RandomStates()
        elif constraint is None:
            return _NoneConstraint()
        elif isinstance(constraint, type):
            return _InstancesOf(constraint)
        elif constraint == "boolean":
            return _Booleans()
        elif constraint == "verbose":
            return _VerboseHelper()
        elif constraint == "cv_object":
            return _CVObjects()
        elif constraint == "nan":
            return _NanConstraint()
        elif callable(constraint):
            return _Callables()
        else:
            raise ValueError(f"Unknown constraint type: {constraint}")

    def validate_params(parameter_constraints, *, prefer_skip_nested_validation):
        """Decorator to validate types and values of functions and methods.

        Parameters
        ----------
        parameter_constraints : dict
            A dictionary `param_name: list of constraints`. See the docstring of
            `validate_parameter_constraints` for a description of the accepted
            constraints.

            Note that the *args and **kwargs parameters are not validated and must not
            be present in the parameter_constraints dictionary.

        prefer_skip_nested_validation : bool
            If True, the validation of parameters of inner estimators or functions
            called by the decorated function will be skipped.

            This is useful to avoid validating many times the parameters passed by the
            user from the public facing API. It's also useful to avoid validating
            parameters that we pass internally to inner functions that are guaranteed to
            be valid by the test suite.

            It should be set to True for most functions, except for those that receive
            non-validated objects as parameters or that are just wrappers around classes
            because they only perform a partial validation.

        Returns
        -------
        decorated_function : function or method
            The decorated function.
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Get the function signature
                sig = signature(func)
            
                # Combine args and kwargs into a single dictionary
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                params = bound_args.arguments

                # Remove self or cls from params if present
                if 'self' in params:
                    del params['self']
                elif 'cls' in params:
                    del params['cls']

                # Validate parameters
                with config_context(skip_parameter_validation=prefer_skip_nested_validation):
                    validate_parameter_constraints(parameter_constraints, params, func.__qualname__)

                return func(*args, **kwargs)
            return wrapper
        return decorator

    class RealNotInt(Real):
        """A type that represents reals that are not instances of int.

        Behaves like float, but also works with values extracted from numpy arrays.
        isintance(1, RealNotInt) -> False
        isinstance(1.0, RealNotInt) -> True
        """
    RealNotInt.register(float)

    def _type_name(t):
        """Convert type into human readable string."""
        module = t.__module__
        qualname = t.__qualname__
        if module == "builtins":
            return qualname
        elif module == "numpy":
            return f"numpy.{qualname}"
        else:
            return f"{module}.{qualname}"

    class _Constraint(ABC):
        """Base class for the constraint objects."""

        def __init__(self):
            self.hidden = False

        @abstractmethod
        def is_satisfied_by(self, val):
            """Whether or not a value satisfies the constraint.

            Parameters
            ----------
            val : object
                The value to check.

            Returns
            -------
            is_satisfied : bool
                Whether or not the constraint is satisfied by this value.
            """
            return isinstance(val, self.type)

        @abstractmethod
        def __str__(self):
            """A human readable representational string of the constraint."""

    class _InstancesOf(_Constraint):
        """Constraint representing instances of a given type.

        Parameters
        ----------
        type : type
            The valid type.
        """

        def __init__(self, type):
            super().__init__()
            self.type = type

        def __str__(self):
            return f'an instance of {_type_name(self.type)!r}'

    class _NoneConstraint(_Constraint):
        """Constraint representing the None singleton."""

        def __str__(self):
            return 'None'

    class _NanConstraint(_Constraint):
        """Constraint representing the indicator `np.nan`."""

        def __str__(self):
            return 'numpy.nan'

    class _PandasNAConstraint(_Constraint):
        """Constraint representing the indicator `pd.NA`."""

        def __str__(self):
            return 'pandas.NA'

    class Options(_Constraint):
        """Constraint representing a finite set of instances of a given type.

        Parameters
        ----------
        type : type

        options : set
            The set of valid scalars.

        deprecated : set or None, default=None
            A subset of the `options` to mark as deprecated in the string
            representation of the constraint.
        """

        def __init__(self, type, options, *, deprecated=None):
            super().__init__()
            self.type = type
            self.options = options
            self.deprecated = deprecated or set()
            if self.deprecated - self.options:
                raise ValueError('The deprecated options must be a subset of the options.')

        def _mark_if_deprecated(self, option):
            """Add a deprecated mark to an option if needed."""
            if option in self.deprecated:
                return f"{option} (deprecated)"
            return str(option)

        def __str__(self):
            options_str = f'{', '.join([self._mark_if_deprecated(o) for o in self.options])}'
            return f'a {_type_name(self.type)} among {{{options_str}}}'

    class StrOptions(Options):
        """Constraint representing a finite set of strings.

        Parameters
        ----------
        options : set of str
            The set of valid strings.

        deprecated : set of str or None, default=None
            A subset of the `options` to mark as deprecated in the string
            representation of the constraint.
        """

        def __init__(self, options, *, deprecated=None):
            super().__init__(type=str, options=options, deprecated=deprecated)

    class Interval(_Constraint):
        """Constraint representing a typed interval.

        Parameters
        ----------
        type : {numbers.Integral, numbers.Real, RealNotInt}
            The set of numbers in which to set the interval.

            If RealNotInt, only reals that don't have the integer type
            are allowed. For example 1.0 is allowed but 1 is not.

        left : float or int or None
            The left bound of the interval. None means left bound is -∞.

        right : float, int or None
            The right bound of the interval. None means right bound is +∞.

        closed : {"left", "right", "both", "neither"}
            Whether the interval is open or closed. Possible choices are:

            - `"left"`: the interval is closed on the left and open on the right.
            It is equivalent to the interval `[ left, right )`.
            - `"right"`: the interval is closed on the right and open on the left.
            It is equivalent to the interval `( left, right ]`.
            - `"both"`: the interval is closed.
            It is equivalent to the interval `[ left, right ]`.
            - `"neither"`: the interval is open.
            It is equivalent to the interval `( left, right )`.

        Notes
        -----
        Setting a bound to `None` and setting the interval closed is valid. For
        instance, strictly speaking, `Interval(Real, 0, None, closed="both")`
        corresponds to `[0, +∞) U {+∞}`.
        """

        def __init__(self, type, left, right, *, closed):
            super().__init__()
            self.type = type
            self.left = left
            self.right = right
            self.closed = closed
            self._check_params()

        def __contains__(self, val):
            if not isinstance(val, Integral) and np.isnan(val):
                return False
            left_cmp = operator.lt if self.closed in ('left', 'both') else operator.le
            right_cmp = operator.gt if self.closed in ('right', 'both') else operator.ge
            left = -np.inf if self.left is None else self.left
            right = np.inf if self.right is None else self.right
            if left_cmp(val, left):
                return False
            if right_cmp(val, right):
                return False
            return True

        def __str__(self):
            type_str = 'an int' if self.type is Integral else 'a float'
            left_bracket = '[' if self.closed in ('left', 'both') else '('
            left_bound = '-inf' if self.left is None else self.left
            right_bound = 'inf' if self.right is None else self.right
            right_bracket = ']' if self.closed in ('right', 'both') else ')'
            if not self.type == Integral and isinstance(self.left, Real):
                left_bound = float(left_bound)
            if not self.type == Integral and isinstance(self.right, Real):
                right_bound = float(right_bound)
            return f'{type_str} in the range {left_bracket}{left_bound}, {right_bound}{right_bracket}'

    class _ArrayLikes(_Constraint):
        """Constraint representing array-likes"""

        def __str__(self):
            return 'an array-like'

    class _SparseMatrices(_Constraint):
        """Constraint representing sparse matrices."""

        def __str__(self):
            return 'a sparse matrix'

    class _Callables(_Constraint):
        """Constraint representing callables."""

        def __str__(self):
            return 'a callable'

    class _RandomStates(_Constraint):
        """Constraint representing random states.

        Convenience class for
        [Interval(Integral, 0, 2**32 - 1, closed="both"), np.random.RandomState, None]
        """

        def __init__(self):
            super().__init__()
            self._constraints = [Interval(Integral, 0, 2 ** 32 - 1, closed='both'), _InstancesOf(np.random.RandomState), _NoneConstraint()]

        def __str__(self):
            return f'{', '.join([str(c) for c in self._constraints[:-1]])} or {self._constraints[-1]}'

    class _Booleans(_Constraint):
        """Constraint representing boolean likes.

        Convenience class for
        [bool, np.bool_, Integral (deprecated)]
        """

        def __init__(self):
            super().__init__()
            self._constraints = [_InstancesOf(bool), _InstancesOf(np.bool_)]

        def __str__(self):
            return f'{', '.join([str(c) for c in self._constraints[:-1]])} or {self._constraints[-1]}'

    class _VerboseHelper(_Constraint):
        """Helper constraint for the verbose parameter.

        Convenience class for
        [Interval(Integral, 0, None, closed="left"), bool, numpy.bool_]
        """

        def __init__(self):
            super().__init__()
            self._constraints = [Interval(Integral, 0, None, closed='left'), _InstancesOf(bool), _InstancesOf(np.bool_)]

        def __str__(self):
            return f'{', '.join([str(c) for c in self._constraints[:-1]])} or {self._constraints[-1]}'

    class MissingValues(_Constraint):
        """Helper constraint for the `missing_values` parameters.

        Convenience for
        [
            Integral,
            Interval(Real, None, None, closed="both"),
            str,   # when numeric_only is False
            None,  # when numeric_only is False
            _NanConstraint(),
            _PandasNAConstraint(),
        ]

        Parameters
        ----------
        numeric_only : bool, default=False
            Whether to consider only numeric missing value markers.

        """

        def __init__(self, numeric_only=False):
            super().__init__()
            self.numeric_only = numeric_only
            self._constraints = [_InstancesOf(Integral), Interval(Real, None, None, closed='both'), _NanConstraint(), _PandasNAConstraint()]
            if not self.numeric_only:
                self._constraints.extend([_InstancesOf(str), _NoneConstraint()])

        def __str__(self):
            return f'{', '.join([str(c) for c in self._constraints[:-1]])} or {self._constraints[-1]}'

    class HasMethods(_Constraint):
        """Constraint representing objects that expose specific methods.

        It is useful for parameters following a protocol and where we don't want to
        impose an affiliation to a specific module or class.

        Parameters
        ----------
        methods : str or list of str
            The method(s) that the object is expected to expose.
        """

        @validate_params({'methods': [str, list]}, prefer_skip_nested_validation=True)
        def __init__(self, methods):
            super().__init__()
            if isinstance(methods, str):
                methods = [methods]
            self.methods = methods

        def __str__(self):
            if len(self.methods) == 1:
                methods = f'{self.methods[0]!r}'
            else:
                methods = f'{', '.join([repr(m) for m in self.methods[:-1]])} and {self.methods[-1]!r}'
            return f'an object implementing {methods}'

    class _IterablesNotString(_Constraint):
        """Constraint representing iterables that are not strings."""

        def __str__(self):
            return 'an iterable'

    class _CVObjects(_Constraint):
        """Constraint representing cv objects.

        Convenient class for
        [
            Interval(Integral, 2, None, closed="left"),
            HasMethods(["split", "get_n_splits"]),
            _IterablesNotString(),
            None,
        ]
        """

        def __init__(self):
            super().__init__()
            self._constraints = [Interval(Integral, 2, None, closed='left'), HasMethods(['split', 'get_n_splits']), _IterablesNotString(), _NoneConstraint()]

        def __str__(self):
            return f'{', '.join([str(c) for c in self._constraints[:-1]])} or {self._constraints[-1]}'

    class Hidden:
        """Class encapsulating a constraint not meant to be exposed to the user.

        Parameters
        ----------
        constraint : str or _Constraint instance
            The constraint to be used internally.
        """

        def __init__(self, constraint):
            self.constraint = constraint

    def generate_invalid_param_val(constraint):
        """Return a value that does not satisfy the constraint.

        Raises a NotImplementedError if there exists no invalid value for this
        constraint.

        This is only useful for testing purpose.

        Parameters
        ----------
        constraint : _Constraint instance
            The constraint to generate a value for.

        Returns
        -------
        val : object
            A value that does not satisfy the constraint.
        """
        if isinstance(constraint, Interval):
            if constraint.type == Integral:
                return constraint.right + 1 if constraint.right is not None else constraint.left - 1
            else:
                return constraint.right + 1.0 if constraint.right is not None else constraint.left - 1.0
        elif isinstance(constraint, _InstancesOf):
            return "invalid_value"  # This will be invalid for most types
        elif isinstance(constraint, Options):
            return "invalid_option"
        elif isinstance(constraint, StrOptions):
            return "invalid_string_option"
        elif isinstance(constraint, _ArrayLikes):
            return 42  # Not array-like
        elif isinstance(constraint, _SparseMatrices):
            return [1, 2, 3]  # Not a sparse matrix
        elif isinstance(constraint, _RandomStates):
            return "not_a_random_state"
        elif isinstance(constraint, _Callables):
            return "not_callable"
        elif isinstance(constraint, _Booleans):
            return "not_a_boolean"
        elif isinstance(constraint, _VerboseHelper):
            return "not_verbose"
        elif isinstance(constraint, _CVObjects):
            return "not_a_cv_object"
        elif isinstance(constraint, _NanConstraint):
            return 0  # Not NaN
        elif isinstance(constraint, MissingValues):
            return complex(1, 1)  # Not a valid missing value
        elif isinstance(constraint, HasMethods):
            return "not_an_object_with_methods"
        else:
            raise NotImplementedError(f"Cannot generate invalid value for {constraint}")

    def generate_valid_param(constraint):
        """Return a value that does satisfy a constraint.

        This is only useful for testing purpose.

        Parameters
        ----------
        constraint : Constraint instance
            The constraint to generate a value for.

        Returns
        -------
        val : object
            A value that does satisfy the constraint.
        """
        if isinstance(constraint, Interval):
            if constraint.type == Integral:
                return constraint.left if constraint.left is not None else (constraint.right - 1)
            else:
                return constraint.left if constraint.left is not None else (constraint.right - 0.5)
        elif isinstance(constraint, _InstancesOf):
            if constraint.type == int:
                return 0
            elif constraint.type == float:
                return 0.0
            elif constraint.type == str:
                return "valid_string"
            else:
                return constraint.type()
        elif isinstance(constraint, Options):
            return next(iter(constraint.options))
        elif isinstance(constraint, StrOptions):
            return next(iter(constraint.options))
        elif isinstance(constraint, _ArrayLikes):
            return np.array([1, 2, 3])
        elif isinstance(constraint, _SparseMatrices):
            return csr_matrix([[1, 2], [3, 4]])
        elif isinstance(constraint, _RandomStates):
            return np.random.RandomState(0)
        elif isinstance(constraint, _Callables):
            return lambda x: x
        elif isinstance(constraint, _Booleans):
            return True
        elif isinstance(constraint, _VerboseHelper):
            return 1
        elif isinstance(constraint, _CVObjects):
            return 3
        elif isinstance(constraint, _NanConstraint):
            return np.nan
        elif isinstance(constraint, MissingValues):
            return np.nan
        elif isinstance(constraint, HasMethods):
            class ValidObject:
                def __getattr__(self, name):
                    return lambda: None
            return ValidObject()
        else:
            raise NotImplementedError(f"Cannot generate valid value for {constraint}")
else:
    from sklearn.utils._param_validation import generate_invalid_param_val
    from sklearn.utils._param_validation import generate_valid_param
    from sklearn.utils._param_validation import validate_parameter_constraints
    from sklearn.utils._param_validation import HasMethods, Hidden, Interval, InvalidParameterError, MissingValues, Options, RealNotInt, StrOptions, _ArrayLikes, _Booleans, _Callables, _CVObjects, _InstancesOf, _IterablesNotString, _NanConstraint, _NoneConstraint, _PandasNAConstraint, _RandomStates, _SparseMatrices, _VerboseHelper, make_constraint, validate_params
