"""Tests for the model training module."""

import numpy as np
from sklearn.pipeline import Pipeline

from src.models.train import create_pipeline, load_model, save_model, train_model


class TestCreatePipeline:
    """Tests for create_pipeline()."""

    def test_returns_pipeline(self):
        pipeline = create_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_has_scaler_and_classifier(self):
        pipeline = create_pipeline()
        step_names = [name for name, _ in pipeline.steps]
        assert "scaler" in step_names
        assert "classifier" in step_names

    def test_custom_parameters(self):
        pipeline = create_pipeline(n_estimators=50, random_state=0)
        clf = pipeline.named_steps["classifier"]
        assert clf.n_estimators == 50
        assert clf.random_state == 0


class TestTrainModel:
    """Tests for train_model()."""

    def test_trains_successfully(self, iris_data):
        model = train_model(iris_data["X_train"], iris_data["y_train"])
        assert isinstance(model, Pipeline)

    def test_model_predicts(self, iris_data):
        model = train_model(iris_data["X_train"], iris_data["y_train"])
        predictions = model.predict(iris_data["X_test"])
        assert len(predictions) == len(iris_data["y_test"])

    def test_model_accuracy_above_threshold(self, iris_data):
        model = train_model(iris_data["X_train"], iris_data["y_train"])
        accuracy = model.score(iris_data["X_test"], iris_data["y_test"])
        assert accuracy > 0.8

    def test_accepts_custom_pipeline(self, iris_data):
        pipeline = create_pipeline(n_estimators=10)
        model = train_model(iris_data["X_train"], iris_data["y_train"], pipeline=pipeline)
        clf = model.named_steps["classifier"]
        assert clf.n_estimators == 10


class TestSaveLoadModel:
    """Tests for save_model() and load_model()."""

    def test_save_and_load(self, iris_data, tmp_path):
        model = train_model(iris_data["X_train"], iris_data["y_train"])
        path = str(tmp_path / "model.joblib")
        save_model(model, path=path)

        loaded = load_model(path=path)
        original_preds = model.predict(iris_data["X_test"])
        loaded_preds = loaded.predict(iris_data["X_test"])
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_load_missing_model_raises(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_model(path=str(tmp_path / "missing.joblib"))
