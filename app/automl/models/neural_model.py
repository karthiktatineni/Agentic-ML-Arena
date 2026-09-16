"""PyTorch Tabular Neural Network Model (PRD v2 Section 35)."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna

from app.automl.models.base_model import BaseAutoMLModel


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_units: List[int], dropout_rate: float, is_classification: bool):
        super(MLP, self).__init__()
        layers = []
        
        in_dim = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h
            
        self.feature_extractor = nn.Sequential(*layers)
        
        if is_classification:
            self.head = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())
        else:
            self.head = nn.Linear(in_dim, 1)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.head(features)


class PyTorchTabularModel(BaseAutoMLModel):
    """PyTorch implementation of a dense Neural Network for tabular data."""

    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        self.model_: Optional[MLP] = None
        self.input_dim_: int = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyperparameters = {}
        
    @property
    def model_name(self) -> str:
        return "pytorch_mlp"
        
    def build_estimator(self, hyperparameters: Dict[str, Any]) -> Any:
        # Since this class itself wraps PyTorch, we just return self
        # However, to be fully compatible with scikit-learn interfaces (if used that way)
        # one would typically return a Skorch wrapper. Here we just return self for HPO.
        return self

    def get_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define HPO search space for PyTorch MLP."""
        return {
            "n_layers": trial.suggest_int("n_layers", 1, 4),
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128, 256, 512]),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "epochs": trial.suggest_int("epochs", 10, 50),
        }

    def train(self, X: pd.DataFrame, y: pd.Series, hyperparameters: Dict[str, Any], **kwargs):
        """Train the PyTorch MLP."""
        self.hyperparameters = hyperparameters
        self.input_dim_ = X.shape[1]
        
        n_layers = hyperparameters.get("n_layers", 2)
        hidden_size = hyperparameters.get("hidden_size", 128)
        hidden_units = [hidden_size] * n_layers
        dropout_rate = hyperparameters.get("dropout_rate", 0.1)
        learning_rate = hyperparameters.get("learning_rate", 1e-3)
        batch_size = hyperparameters.get("batch_size", 64)
        epochs = hyperparameters.get("epochs", 20)
        
        is_classification = self.task_type == "classification"
        
        self.model_ = MLP(
            input_dim=self.input_dim_,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            is_classification=is_classification
        ).to(self.device)
        
        criterion = nn.BCELoss() if is_classification else nn.MSELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=learning_rate)
        
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model_.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict labels or values."""
        preds = self.predict_proba(X)
        if self.task_type == "classification":
            return (preds[:, 1] >= 0.5).astype(int)
        return preds

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict continuous values or probabilities."""
        if self.model_ is None:
            raise ValueError("Model not trained yet.")
            
        self.model_.eval()
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model_(X_tensor).cpu().numpy().flatten()
            
        if self.task_type == "classification":
            # Return binary probability format expected by scikit-learn
            return np.vstack((1 - outputs, outputs)).T
            
        return outputs
