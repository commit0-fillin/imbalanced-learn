"""Class to perform over-sampling using SMOTE and cleaning using Tomek
links."""
import numbers
from sklearn.base import clone
from sklearn.utils import check_X_y
from ..base import BaseSampler
from ..over_sampling import SMOTE
from ..over_sampling.base import BaseOverSampler
from ..under_sampling import TomekLinks
from ..utils import Substitution, check_target_type
from ..utils._docstring import _n_jobs_docstring, _random_state_docstring

@Substitution(sampling_strategy=BaseOverSampler._sampling_strategy_docstring, n_jobs=_n_jobs_docstring, random_state=_random_state_docstring)
class SMOTETomek(BaseSampler):
    """Over-sampling using SMOTE and cleaning using Tomek links.

    Combine over- and under-sampling using SMOTE and Tomek links.

    Read more in the :ref:`User Guide <combine>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    smote : sampler object, default=None
        The :class:`~imblearn.over_sampling.SMOTE` object to use. If not given,
        a :class:`~imblearn.over_sampling.SMOTE` object with default parameters
        will be given.

    tomek : sampler object, default=None
        The :class:`~imblearn.under_sampling.TomekLinks` object to use. If not
        given, a :class:`~imblearn.under_sampling.TomekLinks` object with
        sampling strategy='all' will be given.

    {n_jobs}

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    smote_ : sampler object
        The validated :class:`~imblearn.over_sampling.SMOTE` instance.

    tomek_ : sampler object
        The validated :class:`~imblearn.under_sampling.TomekLinks` instance.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    SMOTEENN : Over-sample using SMOTE followed by under-sampling using Edited
        Nearest Neighbours.

    Notes
    -----
    The method is presented in [1]_.

    Supports multi-class resampling. Refer to SMOTE and TomekLinks regarding
    the scheme which used.

    References
    ----------
    .. [1] G. Batista, B. Bazzan, M. Monard, "Balancing Training Data for
       Automated Annotation of Keywords: a Case Study," In WOB, 10-18, 2003.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.combine import SMOTETomek
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> smt = SMOTETomek(random_state=42)
    >>> X_res, y_res = smt.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})
    """
    _sampling_type = 'over-sampling'
    _parameter_constraints: dict = {**BaseOverSampler._parameter_constraints, 'smote': [SMOTE, None], 'tomek': [TomekLinks, None], 'n_jobs': [numbers.Integral, None]}

    def __init__(self, *, sampling_strategy='auto', random_state=None, smote=None, tomek=None, n_jobs=None):
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.smote = smote
        self.tomek = tomek
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Private function to validate SMOTE and Tomek objects"""
        if self.smote is None:
            self.smote_ = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                k_neighbors=5,
                n_jobs=self.n_jobs,
            )
        elif isinstance(self.smote, SMOTE):
            self.smote_ = clone(self.smote)
        else:
            raise ValueError(
                f"`smote` has to be a `SMOTE` object or None. "
                f"Got {type(self.smote)} instead."
            )

        if self.tomek is None:
            self.tomek_ = TomekLinks(
                sampling_strategy="all", n_jobs=self.n_jobs
            )
        elif isinstance(self.tomek, TomekLinks):
            self.tomek_ = clone(self.tomek)
        else:
            raise ValueError(
                f"`tomek` has to be a `TomekLinks` object or None. "
                f"Got {type(self.tomek)} instead."
            )

        self.smote_.set_params(**{"sampling_strategy": self.sampling_strategy})
