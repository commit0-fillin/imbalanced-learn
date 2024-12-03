"""
The :mod:`imblearn.exceptions` module includes all custom warnings and error
classes and functions used across imbalanced-learn.
"""

def raise_isinstance_error(variable_name, possible_type, variable):
    """Raise consistent error message for isinstance() function.

    Parameters
    ----------
    variable_name : str
        The name of the variable.

    possible_type : type
        The possible type of the variable.

    variable : object
        The variable to check.

    Raises
    ------
    ValueError
        If the instance is not of the possible type.
    """
    if not isinstance(variable, possible_type):
        raise ValueError(
            f"{variable_name} must be an instance of {possible_type.__name__}, "
            f"got {type(variable).__name__} instead."
        )
