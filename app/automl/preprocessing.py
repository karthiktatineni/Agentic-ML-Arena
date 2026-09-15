"""Safe Preprocessing and Out-Of-Fold Target Encoding (PRD v2 Section 13 & 21)."""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler


class TargetTransformer:
    """Applies log1p/expm1 target transformation for right-skewed positive regression targets."""

    SKEWNESS_THRESHOLD = 1.0

    @staticmethod
    def should_transform(y: Union[pd.Series, np.ndarray]) -> bool:
        y_ser = pd.Series(y).dropna()
        if len(y_ser) == 0:
            return False
        if (y_ser <= 0).any():
            return False
        return float(y_ser.skew()) > TargetTransformer.SKEWNESS_THRESHOLD

    @staticmethod
    def wrap(estimator: BaseEstimator, y: Union[pd.Series, np.ndarray]) -> Tuple[Any, bool]:
        """Wrap estimator with TransformedTargetRegressor when skewness warrants log transform."""
        if TargetTransformer.should_transform(y):
            return TransformedTargetRegressor(
                regressor=estimator,
                func=np.log1p,
                inverse_func=np.expm1,
            ), True
        return estimator, False


def get_high_cardinality_categoricals(
    df: pd.DataFrame,
    target_col: str,
    cardinality_threshold: int = 10,
) -> List[str]:
    """Return object/string columns whose unique count exceeds the threshold."""
    cat_cols = [
        c
        for c in df.columns
        if c != target_col and (df[c].dtype == "object" or df[c].dtype.name == "string")
    ]
    return [c for c in cat_cols if df[c].nunique() > cardinality_threshold]


class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Target Encoder with mandatory out-of-fold computation to prevent target leakage."""

    def __init__(self, categorical_cols: List[str], n_splits: int = 5, smoothing: float = 10.0, cv_seed: int = 42):
        self.categorical_cols = categorical_cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.cv_seed = cv_seed
        self.global_target_mean_: float = 0.0
        self.encodings_: Dict[str, Dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit global target encoding mappings for test inference."""
        X_df = X.copy()
        y_ser = pd.Series(y).reset_index(drop=True)
        self.global_target_mean_ = float(y_ser.mean())

        self.encodings_ = {}
        for col in self.categorical_cols:
            if col in X_df.columns:
                counts = X_df.groupby(col).size()
                sums = y_ser.groupby(X_df[col]).sum()
                # Empirical Bayes smoothing formula: (count * mean + smoothing * global_mean) / (count + smoothing)
                smoothed = (sums + self.smoothing * self.global_target_mean_) / (counts + self.smoothing)
                self.encodings_[col] = smoothed.to_dict()

        return self

    def fit_transform_oof(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compute strict out-of-fold target encodings during CV training.

        Each fold's encoding is computed strictly from other folds.
        """
        X_out = X.copy().reset_index(drop=True)
        y_ser = pd.Series(y).reset_index(drop=True)
        self.global_target_mean_ = float(y_ser.mean())

        oof_encoded = {col: np.full(len(X_out), self.global_target_mean_, dtype=float) for col in self.categorical_cols}

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.cv_seed)

        for train_idx, val_idx in kf.split(X_out, y_ser):
            X_tr, y_tr = X_out.iloc[train_idx], y_ser.iloc[train_idx]
            fold_global_mean = float(y_tr.mean())

            for col in self.categorical_cols:
                if col in X_tr.columns:
                    counts = X_tr.groupby(col).size()
                    sums = y_tr.groupby(X_tr[col]).sum()
                    smoothed = (sums + self.smoothing * fold_global_mean) / (counts + self.smoothing)
                    mapping = smoothed.to_dict()

                    # Apply fold mapping to validation rows
                    val_series = X_out.iloc[val_idx][col]
                    oof_encoded[col][val_idx] = val_series.map(mapping).fillna(fold_global_mean)

        for col in self.categorical_cols:
            X_out[f"{col}_target_enc"] = oof_encoded[col]

        # Also fit final mappings for test-set transform
        self.fit(X, y)
        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using global smoothed encodings."""
        X_out = X.copy()
        for col in self.categorical_cols:
            if col in self.encodings_:
                mapping = self.encodings_[col]
                X_out[f"{col}_target_enc"] = X_out[col].map(mapping).fillna(self.global_target_mean_)
        return X_out


SYNONYM_MAPS = {
    "at": "automatic",
    "auto": "automatic",
    "automatic": "automatic",
    "mt": "manual",
    "man": "manual",
    "manual": "manual",
    "diesel": "diesel",
    "petrol": "petrol",
    "cng": "cng",
    "lpg": "lpg",
    "electric": "electric",
    "1st": "1",
    "first": "1",
    "2nd": "2",
    "second": "2",
    "3rd": "3",
    "third": "3",
    "4th": "4",
    "fourth": "4",
}


class StringNormalizer(BaseEstimator, TransformerMixin):
    """Normalize string categoricals (trim whitespace, lowercase, canonical synonyms) to prevent category dilution."""

    def fit(self, X, y=None):
        return self

    def _normalize_val(self, v: Any) -> str:
        if pd.isna(v) or v is None:
            return ""
        s = str(v).strip().lower()
        return SYNONYM_MAPS.get(s, s)

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.apply(lambda col: col.map(self._normalize_val))
        X_arr = np.array(X, dtype=object)
        return np.vectorize(self._normalize_val)(X_arr)


def build_safe_tabular_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaling: str = "robust",
    imputation: str = "median",
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer guaranteed to fit inside CV folds."""
    num_steps = []
    if imputation == "median":
        num_steps.append(("imputer", SimpleImputer(strategy="median")))
    elif imputation == "mean":
        num_steps.append(("imputer", SimpleImputer(strategy="mean")))

    if scaling == "robust":
        num_steps.append(("scaler", RobustScaler()))
    elif scaling == "standard":
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("normalizer", StringNormalizer()),
        ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(num_steps), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline(cat_steps), categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")
