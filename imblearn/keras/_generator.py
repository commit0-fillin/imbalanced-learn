"""Implement generators for ``keras`` which will balance the data."""

def import_keras():
    """Try to import keras from keras and tensorflow.

    This is possible to import the sequence from keras or tensorflow.
    """
    try:
        from keras.utils import Sequence
        return Sequence, True
    except ImportError:
        try:
            from tensorflow.keras.utils import Sequence
            return Sequence, True
        except ImportError:
            return object, False
ParentClass, HAS_KERAS = import_keras()
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.utils import _safe_indexing
from sklearn.utils import check_random_state
from ..tensorflow import balanced_batch_generator as tf_bbg
from ..under_sampling import RandomUnderSampler
from ..utils import Substitution
from ..utils._docstring import _random_state_docstring

class BalancedBatchGenerator(*ParentClass):
    """Create balanced batches when training a keras model.

    Create a keras ``Sequence`` which is given to ``fit``. The
    sampler defines the sampling strategy used to balance the dataset ahead of
    creating the batch. The sampler should have an attribute
    ``sample_indices_``.

    .. versionadded:: 0.4

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Original imbalanced dataset.

    y : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Associated targets.

    sample_weight : ndarray of shape (n_samples,)
        Sample weight.

    sampler : sampler object, default=None
        A sampler instance which has an attribute ``sample_indices_``.
        By default, the sampler used is a
        :class:`~imblearn.under_sampling.RandomUnderSampler`.

    batch_size : int, default=32
        Number of samples per gradient update.

    keep_sparse : bool, default=False
        Either or not to conserve or not the sparsity of the input (i.e. ``X``,
        ``y``, ``sample_weight``). By default, the returned batches will be
        dense.

    random_state : int, RandomState instance or None, default=None
        Control the randomization of the algorithm:

        - If int, ``random_state`` is the seed used by the random number
          generator;
        - If ``RandomState`` instance, random_state is the random number
          generator;
        - If ``None``, the random number generator is the ``RandomState``
          instance used by ``np.random``.

    Attributes
    ----------
    sampler_ : sampler object
        The sampler used to balance the dataset.

    indices_ : ndarray of shape (n_samples, n_features)
        The indices of the samples selected during sampling.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> from imblearn.datasets import make_imbalance
    >>> class_dict = dict()
    >>> class_dict[0] = 30; class_dict[1] = 50; class_dict[2] = 40
    >>> X, y = make_imbalance(iris.data, iris.target, sampling_strategy=class_dict)
    >>> import tensorflow
    >>> y = tensorflow.keras.utils.to_categorical(y, 3)
    >>> model = tensorflow.keras.models.Sequential()
    >>> model.add(
    ...     tensorflow.keras.layers.Dense(
    ...         y.shape[1], input_dim=X.shape[1], activation='softmax'
    ...     )
    ... )
    >>> model.compile(optimizer='sgd', loss='categorical_crossentropy',
    ...               metrics=['accuracy'])
    >>> from imblearn.keras import BalancedBatchGenerator
    >>> from imblearn.under_sampling import NearMiss
    >>> training_generator = BalancedBatchGenerator(
    ...     X, y, sampler=NearMiss(), batch_size=10, random_state=42)
    >>> callback_history = model.fit(training_generator, epochs=10, verbose=0)
    """
    use_sequence_api = True

    def __init__(self, X, y, *, sample_weight=None, sampler=None, batch_size=32, keep_sparse=False, random_state=None):
        if not HAS_KERAS:
            raise ImportError("'No module named 'keras'")
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.sampler = sampler
        self.batch_size = batch_size
        self.keep_sparse = keep_sparse
        self.random_state = random_state
        self._sample()

    def __len__(self):
        return int(self.indices_.size // self.batch_size)

    def __getitem__(self, index):
        X_resampled = _safe_indexing(self.X, self.indices_[index * self.batch_size:(index + 1) * self.batch_size])
        y_resampled = _safe_indexing(self.y, self.indices_[index * self.batch_size:(index + 1) * self.batch_size])
        if issparse(X_resampled) and (not self.keep_sparse):
            X_resampled = X_resampled.toarray()
        if self.sample_weight is not None:
            sample_weight_resampled = _safe_indexing(self.sample_weight, self.indices_[index * self.batch_size:(index + 1) * self.batch_size])
        if self.sample_weight is None:
            return (X_resampled, y_resampled)
        else:
            return (X_resampled, y_resampled, sample_weight_resampled)

@Substitution(random_state=_random_state_docstring)
def balanced_batch_generator(X, y, *, sample_weight=None, sampler=None, batch_size=32, keep_sparse=False, random_state=None):
    """Create a balanced batch generator to train keras model.

    Returns a generator --- as well as the number of step per epoch --- which
    is given to ``fit``. The sampler defines the sampling strategy
    used to balance the dataset ahead of creating the batch. The sampler should
    have an attribute ``sample_indices_``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Original imbalanced dataset.

    y : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Associated targets.

    sample_weight : ndarray of shape (n_samples,), default=None
        Sample weight.

    sampler : sampler object, default=None
        A sampler instance which has an attribute ``sample_indices_``.
        By default, the sampler used is a
        :class:`~imblearn.under_sampling.RandomUnderSampler`.

    batch_size : int, default=32
        Number of samples per gradient update.

    keep_sparse : bool, default=False
        Either or not to conserve or not the sparsity of the input (i.e. ``X``,
        ``y``, ``sample_weight``). By default, the returned batches will be
        dense.

    {random_state}

    Returns
    -------
    generator : generator of tuple
        Generate batch of data. The tuple generated are either (X_batch,
        y_batch) or (X_batch, y_batch, sampler_weight_batch).

    steps_per_epoch : int
        The number of samples per epoch. Required by ``fit_generator`` in
        keras.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> from imblearn.datasets import make_imbalance
    >>> class_dict = dict()
    >>> class_dict[0] = 30; class_dict[1] = 50; class_dict[2] = 40
    >>> from imblearn.datasets import make_imbalance
    >>> X, y = make_imbalance(X, y, sampling_strategy=class_dict)
    >>> import tensorflow
    >>> y = tensorflow.keras.utils.to_categorical(y, 3)
    >>> model = tensorflow.keras.models.Sequential()
    >>> model.add(
    ...     tensorflow.keras.layers.Dense(
    ...         y.shape[1], input_dim=X.shape[1], activation='softmax'
    ...     )
    ... )
    >>> model.compile(optimizer='sgd', loss='categorical_crossentropy',
    ...               metrics=['accuracy'])
    >>> from imblearn.keras import balanced_batch_generator
    >>> from imblearn.under_sampling import NearMiss
    >>> training_generator, steps_per_epoch = balanced_batch_generator(
    ...     X, y, sampler=NearMiss(), batch_size=10, random_state=42)
    >>> callback_history = model.fit(training_generator,
    ...                              steps_per_epoch=steps_per_epoch,
    ...                              epochs=10, verbose=0)
    """
    if sampler is None:
        sampler = RandomUnderSampler(random_state=random_state)
    
    if not hasattr(sampler, 'fit_resample'):
        raise ValueError("'sampler' should have a 'fit_resample' method.")
    
    sampler_ = clone(sampler)
    X_resampled, y_resampled = sampler_.fit_resample(X, y)
    
    if sample_weight is not None:
        sample_weight_resampled = _safe_indexing(sample_weight, sampler_.sample_indices_)
    else:
        sample_weight_resampled = None
    
    return tf_bbg(X_resampled, y_resampled, sample_weight=sample_weight_resampled,
                  batch_size=batch_size, keep_sparse=keep_sparse,
                  random_state=random_state)
