"""Metrics to perform pairwise computation."""
import numbers
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.base import BaseEstimator
from sklearn.utils import check_consistent_length
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted
from ..base import _ParamsValidationMixin
from ..utils._param_validation import StrOptions

class ValueDifferenceMetric(_ParamsValidationMixin, BaseEstimator):
    """Class implementing the Value Difference Metric.

    This metric computes the distance between samples containing only
    categorical features. The distance between feature values of two samples is
    defined as:

    .. math::
       \\delta(x, y) = \\sum_{c=1}^{C} |p(c|x_{f}) - p(c|y_{f})|^{k} \\ ,

    where :math:`x` and :math:`y` are two samples and :math:`f` a given
    feature, :math:`C` is the number of classes, :math:`p(c|x_{f})` is the
    conditional probability that the output class is :math:`c` given that
    the feature value :math:`f` has the value :math:`x` and :math:`k` an
    exponent usually defined to 1 or 2.

    The distance for the feature vectors :math:`X` and :math:`Y` is
    subsequently defined as:

    .. math::
       \\Delta(X, Y) = \\sum_{f=1}^{F} \\delta(X_{f}, Y_{f})^{r} \\ ,

    where :math:`F` is the number of feature and :math:`r` an exponent usually
    defined equal to 1 or 2.

    The definition of this distance was propoed in [1]_.

    Read more in the :ref:`User Guide <vdm>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    n_categories : "auto" or array-like of shape (n_features,), default="auto"
        The number of unique categories per features. If `"auto"`, the number
        of categories will be computed from `X` at `fit`. Otherwise, you can
        provide an array-like of such counts to avoid computation. You can use
        the fitted attribute `categories_` of the
        :class:`~sklearn.preprocesssing.OrdinalEncoder` to deduce these counts.

    k : int, default=1
        Exponent used to compute the distance between feature value.

    r : int, default=2
        Exponent used to compute the distance between the feature vector.

    Attributes
    ----------
    n_categories_ : ndarray of shape (n_features,)
        The number of categories per features.

    proba_per_class_ : list of ndarray of shape (n_categories, n_classes)
        List of length `n_features` containing the conditional probabilities
        for each category given a class.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.10

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    sklearn.neighbors.DistanceMetric : Interface for fast metric computation.

    Notes
    -----
    The input data `X` are expected to be encoded by an
    :class:`~sklearn.preprocessing.OrdinalEncoder` and the data type is used
    should be `np.int32`. If other data types are given, `X` will be converted
    to `np.int32`.

    References
    ----------
    .. [1] Stanfill, Craig, and David Waltz. "Toward memory-based reasoning."
       Communications of the ACM 29.12 (1986): 1213-1228.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array(["green"] * 10 + ["red"] * 10 + ["blue"] * 10).reshape(-1, 1)
    >>> y = [1] * 8 + [0] * 5 + [1] * 7 + [0] * 9 + [1]
    >>> from sklearn.preprocessing import OrdinalEncoder
    >>> encoder = OrdinalEncoder(dtype=np.int32)
    >>> X_encoded = encoder.fit_transform(X)
    >>> from imblearn.metrics.pairwise import ValueDifferenceMetric
    >>> vdm = ValueDifferenceMetric().fit(X_encoded, y)
    >>> pairwise_distance = vdm.pairwise(X_encoded)
    >>> pairwise_distance.shape
    (30, 30)
    >>> X_test = np.array(["green", "red", "blue"]).reshape(-1, 1)
    >>> X_test_encoded = encoder.transform(X_test)
    >>> vdm.pairwise(X_test_encoded)
    array([[0.  ,  0.04,  1.96],
           [0.04,  0.  ,  1.44],
           [1.96,  1.44,  0.  ]])
    """
    _parameter_constraints: dict = {'n_categories': [StrOptions({'auto'}), 'array-like'], 'k': [numbers.Integral], 'r': [numbers.Integral]}

    def __init__(self, *, n_categories='auto', k=1, r=2):
        self.n_categories = n_categories
        self.k = k
        self.r = r

    def fit(self, X, y):
        """Compute the necessary statistics from the training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features), dtype=np.int32
            The input data. The data are expected to be encoded with a
            :class:`~sklearn.preprocessing.OrdinalEncoder`.

        y : ndarray of shape (n_features,)
            The target.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        self._validate_params()
        X, y = self._validate_data(X, y, dtype=np.int32)
        
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        if self.n_categories == "auto":
            self.n_categories_ = np.max(X, axis=0) + 1
        else:
            self.n_categories_ = np.asarray(self.n_categories)
            if self.n_categories_.shape != (n_features,):
                raise ValueError("n_categories must have shape (n_features,)")
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        self.proba_per_class_ = []
        for feature in range(n_features):
            proba = np.zeros((self.n_categories_[feature], n_classes))
            for category in range(self.n_categories_[feature]):
                mask = X[:, feature] == category
                if np.any(mask):
                    proba[category] = np.bincount(y[mask], minlength=n_classes) / np.sum(mask)
            self.proba_per_class_.append(proba)
        
        return self

    def pairwise(self, X, Y=None):
        """Compute the VDM distance pairwise.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features), dtype=np.int32
            The input data. The data are expected to be encoded with a
            :class:`~sklearn.preprocessing.OrdinalEncoder`.

        Y : ndarray of shape (n_samples, n_features), dtype=np.int32
            The input data. The data are expected to be encoded with a
            :class:`~sklearn.preprocessing.OrdinalEncoder`.

        Returns
        -------
        distance_matrix : ndarray of shape (n_samples, n_samples)
            The VDM pairwise distance.
        """
        check_is_fitted(self)
        X = self._validate_data(X, reset=False, dtype=np.int32)
        
        if Y is None:
            Y = X
        else:
            Y = self._validate_data(Y, reset=False, dtype=np.int32)
        
        n_samples_X, n_features = X.shape
        n_samples_Y = Y.shape[0]
        
        if n_features != self.n_features_in_:
            raise ValueError("The number of features in X is different from the number of features used in fit.")
        
        distance_matrix = np.zeros((n_samples_X, n_samples_Y))
        
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                feature_distances = np.zeros(n_features)
                for f in range(n_features):
                    x_val, y_val = X[i, f], Y[j, f]
                    proba_diff = np.abs(self.proba_per_class_[f][x_val] - self.proba_per_class_[f][y_val])
                    feature_distances[f] = np.sum(proba_diff ** self.k)
                distance_matrix[i, j] = np.sum(feature_distances ** self.r) ** (1 / self.r)
        
        return distance_matrix
