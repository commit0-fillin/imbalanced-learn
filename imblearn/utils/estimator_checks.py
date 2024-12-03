"""Utils to check the samplers and compatibility with scikit-learn"""
import re
import sys
import traceback
import warnings
from collections import Counter
from functools import partial
import numpy as np
import pytest
import sklearn
from scipy import sparse
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_blobs, make_classification, make_multilabel_classification
from sklearn.exceptions import SkipTestWarning
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils._tags import _safe_tags
from sklearn.utils._testing import SkipTest, assert_allclose, assert_array_equal, assert_raises_regex, raises, set_random_state
from sklearn.utils.estimator_checks import _enforce_estimator_tags_y, _get_check_estimator_ids, _maybe_mark_xfail
try:
    from sklearn.utils.estimator_checks import _enforce_estimator_tags_x
except ImportError:
    from sklearn.utils.estimator_checks import _enforce_estimator_tags_X as _enforce_estimator_tags_x
from sklearn.utils.fixes import parse_version
from sklearn.utils.multiclass import type_of_target
from imblearn.datasets import make_imbalance
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.under_sampling.base import BaseCleaningSampler, BaseUnderSampler
from imblearn.utils._param_validation import generate_invalid_param_val, make_constraint
sklearn_version = parse_version(sklearn.__version__)

def parametrize_with_checks(estimators):
    """Pytest specific decorator for parametrizing estimator checks.

    The `id` of each check is set to be a pprint version of the estimator
    and the name of the check with its keyword arguments.
    This allows to use `pytest -k` to specify which tests to run::

        pytest test_check_estimators.py -k check_estimators_fit_returns_self

    Parameters
    ----------
    estimators : list of estimators instances
        Estimators to generated checks for.

    Returns
    -------
    decorator : `pytest.mark.parametrize`

    Examples
    --------
    >>> from sklearn.utils.estimator_checks import parametrize_with_checks
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeRegressor

    >>> @parametrize_with_checks([LogisticRegression(),
    ...                           DecisionTreeRegressor()])
    ... def test_sklearn_compatible_estimator(estimator, check):
    ...     check(estimator)
    """
    import pytest
    from sklearn.utils.estimator_checks import check_estimator

    checks = check_estimator(estimators[0], generate_only=True)
    
    def id_func(func):
        if hasattr(func, "__name__"):
            return func.__name__
        else:
            return str(func)

    return pytest.mark.parametrize(
        "estimator,check",
        [(estimator, check) for estimator in estimators for check in checks],
        ids=[
            f"{estimator.__class__.__name__}-{id_func(check)}"
            for estimator in estimators
            for check in checks
        ],
    )
