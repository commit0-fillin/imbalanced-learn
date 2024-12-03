import numpy as np
import pytest
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils._testing import _convert_container
from imblearn.over_sampling import SMOTEN

@pytest.mark.parametrize('sparse_format', ['sparse_csr', 'sparse_csc'])
def test_smoten_sparse_input(data, sparse_format):
    """Check that we handle sparse input in SMOTEN even if it is not efficient.

    Non-regression test for:
    https://github.com/scikit-learn-contrib/imbalanced-learn/issues/971
    """
    X, y = data
    X_sparse = _convert_container(X, constructor_name=sparse_format)
    
    smoten = SMOTEN(random_state=42)
    
    with pytest.warns(UserWarning, match="SMOTEN is not designed to work with sparse input"):
        X_res, y_res = smoten.fit_resample(X_sparse, y)
    
    assert X_res.format == X_sparse.format
    assert X_res.shape[0] > X.shape[0]
    assert y_res.shape[0] == X_res.shape[0]

def test_smoten_categorical_encoder(data):
    """Check that `categorical_encoder` is used when provided."""
    X, y = data
    
    # Create a custom encoder
    custom_encoder = OrdinalEncoder()
    
    smoten_custom = SMOTEN(categorical_encoder=custom_encoder, random_state=42)
    X_res_custom, y_res_custom = smoten_custom.fit_resample(X, y)
    
    # Check if the custom encoder was used
    assert smoten_custom.categorical_encoder_ == custom_encoder
    
    # Compare with default behavior
    smoten_default = SMOTEN(random_state=42)
    X_res_default, y_res_default = smoten_default.fit_resample(X, y)
    
    # Ensure results are different when using custom encoder
    assert not np.array_equal(X_res_custom, X_res_default)
    
    # Check if the resampled data has the expected shape
    assert X_res_custom.shape[0] > X.shape[0]
    assert y_res_custom.shape[0] == X_res_custom.shape[0]
