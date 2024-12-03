"""Utilities for input validation"""
import warnings
from collections import OrderedDict
from functools import wraps
from inspect import Parameter, signature
from numbers import Integral, Real
import numpy as np
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, column_or_1d
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples
from .fixes import _is_pandas_df
SAMPLING_KIND = ('over-sampling', 'under-sampling', 'clean-sampling', 'ensemble', 'bypass')
TARGET_KIND = ('binary', 'multiclass', 'multilabel-indicator')

class ArraysTransformer:
    """A class to convert sampler output arrays to their original types."""

    def __init__(self, X, y):
        self.x_props = self._gets_props(X)
        self.y_props = self._gets_props(y)

def _is_neighbors_object(estimator):
    """Check that the estimator exposes a KNeighborsMixin-like API.

    A KNeighborsMixin-like API exposes the following methods: (i) `kneighbors`,
    (ii) `kneighbors_graph`.

    Parameters
    ----------
    estimator : object
        A scikit-learn compatible estimator.

    Returns
    -------
    is_neighbors_object : bool
        True if the estimator exposes a KNeighborsMixin-like API.
    """
    return (hasattr(estimator, 'kneighbors') and
            hasattr(estimator, 'kneighbors_graph'))

def check_neighbors_object(nn_name, nn_object, additional_neighbor=0):
    """Check the objects is consistent to be a k nearest neighbors.

    Several methods in `imblearn` relies on k nearest neighbors. These objects
    can be passed at initialisation as an integer or as an object that has
    KNeighborsMixin-like attributes. This utility will create or clone said
    object, ensuring it is KNeighbors-like.

    Parameters
    ----------
    nn_name : str
        The name associated to the object to raise an error if needed.

    nn_object : int or KNeighborsMixin
        The object to be checked.

    additional_neighbor : int, default=0
        Sometimes, some algorithm need an additional neighbors.

    Returns
    -------
    nn_object : KNeighborsMixin
        The k-NN object.
    """
    if isinstance(nn_object, int):
        return NearestNeighbors(n_neighbors=nn_object + additional_neighbor)
    elif _is_neighbors_object(nn_object):
        return clone(nn_object)
    else:
        raise ValueError(f"{nn_name} has to be a int or an object with a "
                         "KNeighborsMixin-like API.")

def check_target_type(y, indicate_one_vs_all=False):
    """Check the target types to be conform to the current samplers.

    The current samplers should be compatible with ``'binary'``,
    ``'multilabel-indicator'`` and ``'multiclass'`` targets only.

    Parameters
    ----------
    y : ndarray
        The array containing the target.

    indicate_one_vs_all : bool, default=False
        Either to indicate if the targets are encoded in a one-vs-all fashion.

    Returns
    -------
    y : ndarray
        The returned target.

    is_one_vs_all : bool, optional
        Indicate if the target was originally encoded in a one-vs-all fashion.
        Only returned if ``indicate_multilabel=True``.
    """
    y_type = type_of_target(y)
    if y_type not in TARGET_KIND:
        raise ValueError(f"'y' should be one of {TARGET_KIND}. Got '{y_type}' instead.")

    y_ = np.array(y)
    if y_type == 'binary':
        y_ = column_or_1d(y_)
        classes = np.unique(y_)
        if len(classes) != 2:
            raise ValueError(f"'y' should have exactly 2 classes. Got {len(classes)} classes.")

    if not indicate_one_vs_all:
        return y_
    else:
        return y_, y_type == 'multilabel-indicator'

def _sampling_strategy_all(y, sampling_type):
    """Returns sampling target by targeting all classes."""
    pass

def _sampling_strategy_majority(y, sampling_type):
    """Returns sampling target by targeting the majority class only."""
    pass

def _sampling_strategy_not_majority(y, sampling_type):
    """Returns sampling target by targeting all classes but not the
    majority."""
    pass

def _sampling_strategy_not_minority(y, sampling_type):
    """Returns sampling target by targeting all classes but not the
    minority."""
    pass

def _sampling_strategy_minority(y, sampling_type):
    """Returns sampling target by targeting the minority class only."""
    pass

def _sampling_strategy_auto(y, sampling_type):
    """Returns sampling target auto for over-sampling and not-minority for
    under-sampling."""
    pass

def _sampling_strategy_dict(sampling_strategy, y, sampling_type):
    """Returns sampling target by converting the dictionary depending of the
    sampling."""
    pass

def _sampling_strategy_list(sampling_strategy, y, sampling_type):
    """With cleaning methods, sampling_strategy can be a list to target the
    class of interest."""
    pass

def _sampling_strategy_float(sampling_strategy, y, sampling_type):
    """Take a proportion of the majority (over-sampling) or minority
    (under-sampling) class in binary classification."""
    pass

def check_sampling_strategy(sampling_strategy, y, sampling_type, **kwargs):
    """Sampling target validation for samplers.

    Checks that ``sampling_strategy`` is of consistent type and return a
    dictionary containing each targeted class with its corresponding
    number of sample. It is used in :class:`~imblearn.base.BaseSampler`.

    Parameters
    ----------
    sampling_strategy : float, str, dict, list or callable,
        Sampling information to sample the data set.

        - When ``float``:

            For **under-sampling methods**, it corresponds to the ratio
            :math:`\\alpha_{us}` defined by :math:`N_{rM} = \\alpha_{us}
            \\times N_{m}` where :math:`N_{rM}` and :math:`N_{m}` are the
            number of samples in the majority class after resampling and the
            number of samples in the minority class, respectively;

            For **over-sampling methods**, it correspond to the ratio
            :math:`\\alpha_{os}` defined by :math:`N_{rm} = \\alpha_{os}
            \\times N_{m}` where :math:`N_{rm}` and :math:`N_{M}` are the
            number of samples in the minority class after resampling and the
            number of samples in the majority class, respectively.

            .. warning::
               ``float`` is only available for **binary** classification. An
               error is raised for multi-class classification and with cleaning
               samplers.

        - When ``str``, specify the class targeted by the resampling. For
          **under- and over-sampling methods**, the number of samples in the
          different classes will be equalized. For **cleaning methods**, the
          number of samples will not be equal. Possible choices are:

            ``'minority'``: resample only the minority class;

            ``'majority'``: resample only the majority class;

            ``'not minority'``: resample all classes but the minority class;

            ``'not majority'``: resample all classes but the majority class;

            ``'all'``: resample all classes;

            ``'auto'``: for under-sampling methods, equivalent to ``'not
            minority'`` and for over-sampling methods, equivalent to ``'not
            majority'``.

        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.

          .. warning::
             ``dict`` is available for both **under- and over-sampling
             methods**. An error is raised with **cleaning methods**. Use a
             ``list`` instead.

        - When ``list``, the list contains the targeted classes. It used only
          for **cleaning methods**.

          .. warning::
             ``list`` is available for **cleaning methods**. An error is raised
             with **under- and over-sampling methods**.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.

    y : ndarray of shape (n_samples,)
        The target array.

    sampling_type : {{'over-sampling', 'under-sampling', 'clean-sampling'}}
        The type of sampling. Can be either ``'over-sampling'``,
        ``'under-sampling'``, or ``'clean-sampling'``.

    **kwargs : dict
        Dictionary of additional keyword arguments to pass to
        ``sampling_strategy`` when this is a callable.

    Returns
    -------
    sampling_strategy_converted : dict
        The converted and validated sampling target. Returns a dictionary with
        the key being the class target and the value being the desired
        number of samples.
    """
    pass
SAMPLING_TARGET_KIND = {'minority': _sampling_strategy_minority, 'majority': _sampling_strategy_majority, 'not minority': _sampling_strategy_not_minority, 'not majority': _sampling_strategy_not_majority, 'all': _sampling_strategy_all, 'auto': _sampling_strategy_auto}

def _deprecate_positional_args(f):
    """Decorator for methods that issues warnings for positional arguments

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Parameters
    ----------
    f : function
        function to check arguments on.
    """
    pass

def _check_X(X):
    """Check X and do not check it if a dataframe."""
    pass
