"""Class to perform under-sampling by removing Tomek's links."""
import numbers
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing
from ...utils import Substitution
from ...utils._docstring import _n_jobs_docstring
from ..base import BaseCleaningSampler

@Substitution(sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring, n_jobs=_n_jobs_docstring)
class TomekLinks(BaseCleaningSampler):
    """Under-sampling by removing Tomek's links.

    Read more in the :ref:`User Guide <tomek_links>`.

    Parameters
    ----------
    {sampling_strategy}

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

        .. versionadded:: 0.4

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    EditedNearestNeighbours : Undersample by samples edition.

    CondensedNearestNeighbour : Undersample by samples condensation.

    RandomUnderSampler : Randomly under-sample the dataset.

    Notes
    -----
    This method is based on [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    References
    ----------
    .. [1] I. Tomek, "Two modifications of CNN," In Systems, Man, and
       Cybernetics, IEEE Transactions on, vol. 6, pp 769-772, 1976.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import TomekLinks
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> tl = TomekLinks()
    >>> X_res, y_res = tl.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 897, 0: 100}})
    """
    _parameter_constraints: dict = {**BaseCleaningSampler._parameter_constraints, 'n_jobs': [numbers.Integral, None]}

    def __init__(self, *, sampling_strategy='auto', n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_jobs = n_jobs

    @staticmethod
    def is_tomek(y, nn_index, class_type):
        """Detect if samples are Tomek's link.

        More precisely, it uses the target vector and the first neighbour of
        every sample point and looks for Tomek pairs. Returning a boolean
        vector with True for majority Tomek links.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target vector of the data set, necessary to keep track of whether a
            sample belongs to minority or not.

        nn_index : ndarray of shape (len(y),)
            The index of the closes nearest neighbour to a sample point.

        class_type : int or str
            The label of the minority class.

        Returns
        -------
        is_tomek : ndarray of shape (len(y), )
            Boolean vector on len( # samples ), with True for majority samples
            that are Tomek links.
        """
        is_tomek = np.zeros(len(y), dtype=bool)
        
        for index in range(len(y)):
            if y[index] != class_type:
                nn = nn_index[index]
                if y[nn] == class_type and nn_index[nn] == index:
                    is_tomek[index] = True
        
        return is_tomek
