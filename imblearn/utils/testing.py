"""Test utilities."""
import inspect
import pkgutil
from importlib import import_module
from operator import itemgetter
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.neighbors import KDTree
from sklearn.utils._testing import ignore_warnings

def all_estimators(type_filter=None):
    """Get a list of all estimators from imblearn.

    This function crawls the module and gets all classes that inherit
    from BaseEstimator. Classes that are defined in test-modules are not
    included.
    By default meta_estimators are also not included.
    This function is adapted from sklearn.

    Parameters
    ----------
    type_filter : str, list of str, or None, default=None
        Which kind of estimators should be returned. If None, no
        filter is applied and all estimators are returned.  Possible
        values are 'sampler' to get estimators only of these specific
        types, or a list of these to get the estimators that fit at
        least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.
    """
    import imblearn
    from imblearn.base import BaseSampler

    def is_abstract(c):
        if not(hasattr(c, '__abstractmethods__')):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    modules_to_ignore = {'tests', 'utils'}
    root = str(Path(imblearn.__file__).parent)

    for importer, modname, ispkg in pkgutil.walk_packages(path=[root],
                                                          prefix='imblearn.'):
        mod_parts = modname.split('.')
        if any(part in modules_to_ignore for part in mod_parts):
            continue
        module = import_module(modname)
        classes = inspect.getmembers(module, inspect.isclass)
        all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [c for c in all_classes
                  if (issubclass(c[1], BaseSampler) and
                      c[0] != 'BaseSampler' and
                      not is_abstract(c[1]))]

    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        estimators = [est for est in estimators
                      if any(hasattr(est[1], attr)
                             for attr in type_filter)]

    return sorted(estimators, key=itemgetter(0))

class _CustomNearestNeighbors(BaseEstimator):
    """Basic implementation of nearest neighbors not relying on scikit-learn.

    `kneighbors_graph` is ignored and `metric` does not have any impact.
    """

    def __init__(self, n_neighbors=1, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        """This method is not used within imblearn but it is required for
        duck-typing."""
        if X is None:
            raise ValueError("X must be provided")
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        
        if mode not in ['connectivity', 'distance']:
            raise ValueError("Unsupported mode, must be 'connectivity' or 'distance'")
        
        # Use KDTree for efficient nearest neighbor search
        tree = KDTree(X)
        distances, indices = tree.query(X, k=n_neighbors + 1)  # +1 because the first neighbor is the point itself
        
        # Create the graph
        if mode == 'connectivity':
            graph = sparse.lil_matrix((n_samples, n_samples), dtype=int)
            for i in range(n_samples):
                graph[i, indices[i, 1:]] = 1  # Skip the first neighbor (self)
        else:  # mode == 'distance'
            graph = sparse.lil_matrix((n_samples, n_samples), dtype=float)
            for i in range(n_samples):
                graph[i, indices[i, 1:]] = distances[i, 1:]  # Skip the first neighbor (self)
        
        return graph.tocsr()

class _CustomClusterer(BaseEstimator):
    """Class that mimics a cluster that does not expose `cluster_centers_`."""

    def __init__(self, n_clusters=1, expose_cluster_centers=True):
        self.n_clusters = n_clusters
        self.expose_cluster_centers = expose_cluster_centers
