"""Tests for the data loading module."""

import numpy as np
import pandas as pd

from src.data.loader import export_sample_data, load_data, load_data_as_dataframe


class TestLoadData:
    """Tests for load_data()."""

    def test_returns_correct_number_of_elements(self):
        result = load_data()
        assert len(result) == 6

    def test_train_test_shapes(self):
        X_train, X_test, y_train, y_test, _, _ = load_data()
        assert X_train.shape[0] + X_test.shape[0] == 150
        assert X_train.shape[1] == 4
        assert X_test.shape[1] == 4
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]

    def test_default_split_ratio(self):
        X_train, X_test, _, _, _, _ = load_data()
        total = X_train.shape[0] + X_test.shape[0]
        assert X_test.shape[0] == int(total * 0.2)

    def test_custom_split_ratio(self):
        X_train, X_test, _, _, _, _ = load_data(test_size=0.3, random_state=42)
        total = X_train.shape[0] + X_test.shape[0]
        assert X_test.shape[0] == int(total * 0.3)

    def test_feature_names(self):
        _, _, _, _, feature_names, _ = load_data()
        assert len(feature_names) == 4
        assert "sepal length (cm)" in feature_names

    def test_target_names(self):
        _, _, _, _, _, target_names = load_data()
        assert len(target_names) == 3
        assert "setosa" in target_names

    def test_data_types(self):
        X_train, X_test, y_train, y_test, _, _ = load_data()
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)

    def test_reproducibility(self):
        result1 = load_data(random_state=42)
        result2 = load_data(random_state=42)
        np.testing.assert_array_equal(result1[0], result2[0])


class TestLoadDataAsDataFrame:
    """Tests for load_data_as_dataframe()."""

    def test_returns_dataframe_and_series(self):
        features, target = load_data_as_dataframe()
        assert isinstance(features, pd.DataFrame)
        assert isinstance(target, pd.Series)

    def test_correct_shape(self):
        features, target = load_data_as_dataframe()
        assert features.shape == (150, 4)
        assert len(target) == 150


class TestExportSampleData:
    """Tests for export_sample_data()."""

    def test_exports_csv(self, tmp_path):
        output_dir = str(tmp_path / "sample")
        csv_path = export_sample_data(output_dir)
        df = pd.read_csv(csv_path)
        assert len(df) == 150
        assert "target" in df.columns
