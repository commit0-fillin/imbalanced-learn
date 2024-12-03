"""Implement generators for ``tensorflow`` which will balance the data."""
from scipy.sparse import issparse
from sklearn.base import clone
from sklearn.utils import _safe_indexing, check_random_state
from ..under_sampling import RandomUnderSampler
from ..utils import Substitution
from ..utils._docstring import _random_state_docstring

@Substitution(random_state=_random_state_docstring)
def balanced_batch_generator(X, y, *, sample_weight=None, sampler=None, batch_size=32, keep_sparse=False, random_state=None):
    """Create a balanced batch generator to train tensorflow model.

    Returns a generator --- as well as the number of step per epoch --- to
    iterate to get the mini-batches. The sampler defines the sampling strategy
    used to balance the dataset ahead of creating the batch. The sampler should
    have an attribute ``sample_indices_``.

    .. versionadded:: 0.4

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
        Either or not to conserve or not the sparsity of the input ``X``. By
        default, the returned batches will be dense.

    {random_state}

    Returns
    -------
    generator : generator of tuple
        Generate batch of data. The tuple generated are either (X_batch,
        y_batch) or (X_batch, y_batch, sampler_weight_batch).

    steps_per_epoch : int
        The number of samples per epoch.
    """
    random_state = check_random_state(random_state)
    
    if sampler is None:
        sampler = RandomUnderSampler(random_state=random_state)
    else:
        sampler = clone(sampler)
        if random_state is not None:
            sampler.set_params(random_state=random_state)
    
    # Fit the sampler
    sampler.fit_resample(X, y)
    
    # Get the indices of the balanced set
    indices = sampler.sample_indices_
    
    # Calculate steps per epoch
    steps_per_epoch = len(indices) // batch_size
    
    def generator():
        while True:
            # Shuffle indices
            random_state.shuffle(indices)
            
            for start in range(0, len(indices), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                X_batch = _safe_indexing(X, batch_indices)
                y_batch = _safe_indexing(y, batch_indices)
                
                if not keep_sparse and issparse(X_batch):
                    X_batch = X_batch.toarray()
                
                if sample_weight is not None:
                    sw_batch = _safe_indexing(sample_weight, batch_indices)
                    yield X_batch, y_batch, sw_batch
                else:
                    yield X_batch, y_batch
    
    return generator(), steps_per_epoch
