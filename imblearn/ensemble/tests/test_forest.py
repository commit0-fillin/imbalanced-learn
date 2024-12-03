import numpy as np
import pytest
import sklearn
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import parse_version
from imblearn.ensemble import BalancedRandomForestClassifier
sklearn_version = parse_version(sklearn.__version__)

def test_balanced_bagging_classifier_n_features():
    """Check that we raise a FutureWarning when accessing `n_features_`."""
    X, y = make_classification(n_samples=100, n_features=4, n_informative=2, n_redundant=0, n_classes=2, random_state=0)
    clf = BalancedRandomForestClassifier(n_estimators=5, random_state=0)
    clf.fit(X, y)
    
    with pytest.warns(FutureWarning, match="`n_features_` is deprecated"):
        _ = clf.n_features_

def test_balanced_random_forest_change_behaviour(imbalanced_dataset):
    """Check that we raise a change of behaviour for the parameters `sampling_strategy`
    and `replacement`.
    """
    X, y = imbalanced_dataset
    clf = BalancedRandomForestClassifier(n_estimators=5, random_state=0)
    
    with pytest.warns(FutureWarning, match="The default value of `sampling_strategy` will change"):
        clf.fit(X, y)
    
    with pytest.warns(FutureWarning, match="The default value of `replacement` will change"):
        BalancedRandomForestClassifier(n_estimators=5, random_state=0, sampling_strategy='auto').fit(X, y)

@pytest.mark.skipif(parse_version(sklearn_version.base_version) < parse_version('1.4'), reason='scikit-learn should be >= 1.4')
def test_missing_values_is_resilient():
    """Check that forest can deal with missing values and has decent performance."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # Introduce missing values
    rng = np.random.RandomState(0)
    mask = rng.binomial(1, 0.1, X_train.shape).astype(bool)
    X_train[mask] = np.nan
    
    clf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    assert score > 0.7, f"Poor performance with missing values: {score:.3f}"

@pytest.mark.skipif(parse_version(sklearn_version.base_version) < parse_version('1.4'), reason='scikit-learn should be >= 1.4')
def test_missing_value_is_predictive():
    """Check that the forest learns when missing values are only present for
    a predictive feature."""
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, n_classes=2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # Make the first feature predictive of the target by setting it to 1 when y is 1
    X_train[:, 0] = (y_train == 1).astype(float)
    X_test[:, 0] = (y_test == 1).astype(float)
    
    # Introduce missing values in the first feature
    mask = np.random.RandomState(0).binomial(1, 0.5, X_train.shape[0]).astype(bool)
    X_train[mask, 0] = np.nan
    
    clf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    assert score > 0.8, f"Poor performance with predictive missing values: {score:.3f}"
    
    # Check feature importances
    importances = clf.feature_importances_
    assert importances[0] > 0.3, f"First feature should be important, got {importances[0]:.3f}"
