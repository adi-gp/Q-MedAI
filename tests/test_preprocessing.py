import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from src.data.loader import load_dataset
from src.preprocessing.pipeline import LeakageSafePreprocessor, split_and_preprocess


def test_select_k_best_explicitly_uses_f_classif_and_requested_count():
    bundle = load_dataset()
    prepared = split_and_preprocess(bundle.X, bundle.y, method="SelectKBest", n_features=6)
    assert prepared.X_train.shape[1] == 6
    assert prepared.X_test.shape[1] == 6
    assert prepared.preprocessor.reducer.score_func is f_classif
    assert len(prepared.preprocessor.feature_names_out_) == 6


def test_pca_and_all_supported_feature_counts_work():
    bundle = load_dataset()
    for count in (4, 6, 8):
        prepared = split_and_preprocess(bundle.X, bundle.y, method="PCA", n_features=count)
        assert prepared.X_train.shape[1] == count
        assert prepared.preprocessor.feature_names_out_ == [
            f"PCA component {index + 1}" for index in range(count)
        ]


def test_preprocessing_fits_only_training_data_and_transform_never_refits():
    # The held-out values are intentionally extreme; fitting on all rows would change the mean.
    X_train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0, 4.0]})
    y_train = np.array([0, 0, 1, 1])
    X_held_out = pd.DataFrame({"a": [1000.0, 1001.0], "b": [2000.0, 2001.0]})
    preprocessor = LeakageSafePreprocessor(method="PCA", n_features=2).fit(X_train, y_train)
    fitted_mean = preprocessor.scaler.mean_.copy()
    transformed = preprocessor.transform(X_held_out)
    combined_mean = pd.concat([X_train, X_held_out]).mean().to_numpy()

    assert transformed.shape == (2, 2)
    assert np.allclose(fitted_mean, X_train.mean().to_numpy())
    assert not np.allclose(fitted_mean, combined_mean)
    # A test transform must not mutate fitted training-only statistics.
    assert np.allclose(preprocessor.scaler.mean_, fitted_mean)
