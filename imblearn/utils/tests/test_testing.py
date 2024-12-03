"""Test for the testing module"""
import numpy as np
import pytest
from sklearn.neighbors._base import KNeighborsMixin
from imblearn.base import SamplerMixin
from imblearn.utils.testing import _CustomNearestNeighbors, all_estimators

def test_custom_nearest_neighbors():
    """Check that our custom nearest neighbors can be used for our internal
    duck-typing."""
    from imblearn.utils.testing import _CustomNearestNeighbors
    from sklearn.neighbors._base import KNeighborsMixin

    custom_nn = _CustomNearestNeighbors()
    
    # Check if _CustomNearestNeighbors has the required methods for duck-typing
    assert hasattr(custom_nn, 'kneighbors_graph')
    
    # Check if _CustomNearestNeighbors is not an instance of KNeighborsMixin
    assert not isinstance(custom_nn, KNeighborsMixin)
    
    # Test the kneighbors_graph method
    import numpy as np
    X = np.array([[0, 0], [1, 1], [2, 2]])
    graph = custom_nn.kneighbors_graph(X)
    
    # Check if the graph is a sparse matrix
    from scipy import sparse
    assert sparse.issparse(graph)
    
    # Check the shape of the graph
    assert graph.shape == (3, 3)
    
    # Check if the diagonal is zero (a point is not its own neighbor)
    assert np.all(graph.diagonal() == 0)
