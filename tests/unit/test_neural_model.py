"""Unit tests for PyTorch Tabular Model."""

import pytest

pytest.importorskip("torch")

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from app.automl.models.neural_model import PyTorchTabularModel

def test_pytorch_classification():
    X_np, y_np = make_classification(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X_np)
    y = pd.Series(y_np)
    
    model = PyTorchTabularModel(task_type="classification")
    hyperparams = {
        "n_layers": 1,
        "hidden_size": 32,
        "dropout_rate": 0.0,
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 2
    }
    
    model.train(X, y, hyperparams)
    
    preds = model.predict(X)
    assert preds.shape == (100,)
    assert set(np.unique(preds)).issubset({0, 1})
    
    probas = model.predict_proba(X)
    assert probas.shape == (100, 2)
    assert np.all((probas >= 0) & (probas <= 1))
    np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)

def test_pytorch_regression():
    X_np, y_np = make_regression(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X_np)
    y = pd.Series(y_np)
    
    model = PyTorchTabularModel(task_type="regression")
    hyperparams = {
        "n_layers": 1,
        "hidden_size": 32,
        "dropout_rate": 0.0,
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 2
    }
    
    model.train(X, y, hyperparams)
    
    preds = model.predict(X)
    assert preds.shape == (100,)
