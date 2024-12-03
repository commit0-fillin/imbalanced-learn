"""Test the module SMOTENC."""
from collections import Counter
import numpy as np
import pytest
import sklearn
from scipy import sparse
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import parse_version
from imblearn.over_sampling import SMOTENC
from imblearn.utils.estimator_checks import _set_checking_parameters, check_param_validation
sklearn_version = parse_version(sklearn.__version__)

def test_smotenc_categorical_encoder():
    """Check that we can pass our own categorical encoder."""
    X, y = make_classification(n_samples=100, n_features=5, n_informative=2,
                               n_redundant=2, n_repeated=0, n_classes=2,
                               n_clusters_per_class=1, weights=[0.9, 0.1],
                               class_sep=0.8, random_state=0)
    X[:, [0, 2]] = X[:, [0, 2]].astype(int)
    categorical_features = [0, 2]
    
    custom_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    smote_nc = SMOTENC(categorical_features=categorical_features,
                       categorical_encoder=custom_encoder,
                       random_state=0)
    
    X_resampled, y_resampled = smote_nc.fit_resample(X, y)
    
    assert isinstance(smote_nc.categorical_encoder_, OneHotEncoder)
    assert smote_nc.categorical_encoder_ is custom_encoder

def test_smotenc_deprecation_ohe_():
    """Check that we raise a deprecation warning when using `ohe_`."""
    X, y = make_classification(n_samples=100, n_features=5, n_informative=2,
                               n_redundant=2, n_repeated=0, n_classes=2,
                               n_clusters_per_class=1, weights=[0.9, 0.1],
                               class_sep=0.8, random_state=0)
    X[:, [0, 2]] = X[:, [0, 2]].astype(int)
    categorical_features = [0, 2]
    
    smote_nc = SMOTENC(categorical_features=categorical_features, random_state=0)
    smote_nc.fit_resample(X, y)
    
    with pytest.warns(FutureWarning, match="The attribute `ohe_` is deprecated"):
        _ = smote_nc.ohe_

def test_smotenc_param_validation():
    """Check that we validate the parameters correctly since this estimator requires
    a specific parameter.
    """
    X, y = make_classification(n_samples=100, n_features=5, n_informative=2,
                               n_redundant=2, n_repeated=0, n_classes=2,
                               n_clusters_per_class=1, weights=[0.9, 0.1],
                               class_sep=0.8, random_state=0)
    X[:, [0, 2]] = X[:, [0, 2]].astype(int)
    
    smote_nc = SMOTENC(categorical_features=[0, 2])
    check_param_validation(smote_nc, X, y)
    
    with pytest.raises(ValueError, match="The parameter 'categorical_features' should"):
        SMOTENC()

def test_smotenc_bool_categorical():
    """Check that we don't try to early convert the full input data to numeric when
    handling a pandas dataframe.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/issues/974
    """
    pd = pytest.importorskip("pandas")
    
    X = pd.DataFrame({
        "num": [1, 2, 3, 4, 5],
        "cat": ["A", "B", "C", "A", "B"],
        "bool": [True, False, True, False, True]
    })
    y = pd.Series([0, 0, 0, 1, 1])
    
    smote_nc = SMOTENC(categorical_features=[1, 2], random_state=42)
    X_res, y_res = smote_nc.fit_resample(X, y)
    
    assert isinstance(X_res, pd.DataFrame)
    assert X_res["bool"].dtype == bool
    assert X_res["cat"].dtype == object
    assert X_res["num"].dtype == int

def test_smotenc_categorical_features_str():
    """Check that we support array-like of strings for `categorical_features` using
    pandas dataframe.
    """
    pd = pytest.importorskip("pandas")
    
    X = pd.DataFrame({
        "num": [1, 2, 3, 4, 5],
        "cat": ["A", "B", "C", "A", "B"],
        "bool": [True, False, True, False, True]
    })
    y = pd.Series([0, 0, 0, 1, 1])
    
    smote_nc = SMOTENC(categorical_features=["cat", "bool"], random_state=42)
    X_res, y_res = smote_nc.fit_resample(X, y)
    
    assert isinstance(X_res, pd.DataFrame)
    assert X_res["bool"].dtype == bool
    assert X_res["cat"].dtype == object
    assert X_res["num"].dtype == int

def test_smotenc_categorical_features_auto():
    """Check that we can automatically detect categorical features based on pandas
    dataframe.
    """
    pd = pytest.importorskip("pandas")
    
    X = pd.DataFrame({
        "num": [1, 2, 3, 4, 5],
        "cat": pd.Categorical(["A", "B", "C", "A", "B"]),
        "bool": [True, False, True, False, True]
    })
    y = pd.Series([0, 0, 0, 1, 1])
    
    smote_nc = SMOTENC(categorical_features="auto", random_state=42)
    X_res, y_res = smote_nc.fit_resample(X, y)
    
    assert isinstance(X_res, pd.DataFrame)
    assert X_res["bool"].dtype == bool
    assert X_res["cat"].dtype == pd.CategoricalDtype(categories=["A", "B", "C"])
    assert X_res["num"].dtype == int
    assert_array_equal(smote_nc.categorical_features_, [1, 2])

def test_smote_nc_categorical_features_auto_error():
    """Check that we raise a proper error when we cannot use the `'auto'` mode."""
    X, y = make_classification(n_samples=100, n_features=5, n_informative=2,
                               n_redundant=2, n_repeated=0, n_classes=2,
                               n_clusters_per_class=1, weights=[0.9, 0.1],
                               class_sep=0.8, random_state=0)
    
    smote_nc = SMOTENC(categorical_features="auto", random_state=42)
    
    with pytest.raises(ValueError, match="The 'auto' option for `categorical_features`"):
        smote_nc.fit_resample(X, y)
