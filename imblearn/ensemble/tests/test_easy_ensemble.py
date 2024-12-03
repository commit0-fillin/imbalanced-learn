"""Test the module easy ensemble."""
import numpy as np
import pytest
import sklearn
from sklearn.datasets import load_iris, make_hastie_10_2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import parse_version
from imblearn.datasets import make_imbalance
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler
sklearn_version = parse_version(sklearn.__version__)
iris = load_iris()
RND_SEED = 0
X = np.array([[0.5220963, 0.11349303], [0.59091459, 0.40692742], [1.10915364, 0.05718352], [0.22039505, 0.26469445], [1.35269503, 0.44812421], [0.85117925, 1.0185556], [-2.10724436, 0.70263997], [-0.23627356, 0.30254174], [-1.23195149, 0.15427291], [-0.58539673, 0.62515052]])
Y = np.array([1, 2, 2, 2, 1, 0, 1, 1, 1, 0])

def test_easy_ensemble_classifier_n_features():
    """Check that we raise a FutureWarning when accessing `n_features_`."""
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=RND_SEED,
    )
    eec = EasyEnsembleClassifier(random_state=RND_SEED, n_estimators=5)
    eec.fit(X, y)

    with pytest.warns(FutureWarning, match="`n_features_` attribute is deprecated"):
        _ = eec.n_features_

    # Check that n_features_in_ is set
    assert hasattr(eec, "n_features_in_")
    assert eec.n_features_in_ == X.shape[1]
