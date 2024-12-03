"""Test the module edited nearest neighbour."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_array_equal
from imblearn.under_sampling import EditedNearestNeighbours
X = np.array([[2.59928271, 0.93323465], [0.25738379, 0.95564169], [1.42772181, 0.526027], [1.92365863, 0.82718767], [-0.10903849, -0.12085181], [-0.284881, -0.62730973], [0.57062627, 1.19528323], [0.03394306, 0.03986753], [0.78318102, 2.59153329], [0.35831463, 1.33483198], [-0.14313184, -1.0412815], [0.01936241, 0.17799828], [-1.25020462, -0.40402054], [-0.09816301, -0.74662486], [-0.01252787, 0.34102657], [0.52726792, -0.38735648], [0.2821046, -0.07862747], [0.05230552, 0.09043907], [0.15198585, 0.12512646], [0.70524765, 0.39816382]])
Y = np.array([1, 2, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 1, 2, 1])

def test_enn_check_kind_selection():
    """Check that `check_sel="all"` is more conservative than
    `check_sel="mode"`."""
    X, y = make_classification(
        n_samples=200, n_classes=2, weights=[0.1, 0.9], random_state=42
    )

    enn_all = EditedNearestNeighbours(kind_sel="all", random_state=42)
    enn_mode = EditedNearestNeighbours(kind_sel="mode", random_state=42)

    X_resampled_all, y_resampled_all = enn_all.fit_resample(X, y)
    X_resampled_mode, y_resampled_mode = enn_mode.fit_resample(X, y)

    # Check that "all" selection is more conservative (removes fewer samples)
    assert len(X_resampled_all) >= len(X_resampled_mode)
    assert len(y_resampled_all) >= len(y_resampled_mode)

    # Check that the samples in "all" are a superset of samples in "mode"
    assert set(map(tuple, X_resampled_mode)).issubset(set(map(tuple, X_resampled_all)))
    assert set(y_resampled_mode).issubset(set(y_resampled_all))

    # Check that both methods have reduced the number of samples
    assert len(X_resampled_all) < len(X)
    assert len(X_resampled_mode) < len(X)

    # Check that the class distribution has been improved (more balanced)
    original_ratio = np.sum(y == 0) / np.sum(y == 1)
    all_ratio = np.sum(y_resampled_all == 0) / np.sum(y_resampled_all == 1)
    mode_ratio = np.sum(y_resampled_mode == 0) / np.sum(y_resampled_mode == 1)

    assert abs(1 - all_ratio) < abs(1 - original_ratio)
    assert abs(1 - mode_ratio) < abs(1 - original_ratio)
