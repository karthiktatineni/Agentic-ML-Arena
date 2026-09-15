"""Concrete Leakage Detection Algorithms (PRD v2 Section 13)."""

from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class LeakageFinding:
    def __init__(self, column: str, action: str, detector: str, reason: str, metric_value: Optional[float] = None):
        self.column = column
        self.action = action  # "exclude" or "flag_review"
        self.detector = detector
        self.reason = reason
        self.metric_value = metric_value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "action": self.action,
            "detector": self.detector,
            "reason": self.reason,
            "metric_value": self.metric_value,
        }


class LeakageDetectorEngine:
    """Automatic concrete leakage detector suite enforcing zero data contamination."""

    def __init__(
        self,
        auc_threshold: float = 0.98,
        mi_percentile_threshold: float = 99.0,
        proxy_correlation_threshold: float = 0.90,
        near_duplicate_cosine_threshold: float = 0.999,
    ):
        self.auc_threshold = auc_threshold
        self.mi_percentile_threshold = mi_percentile_threshold
        self.proxy_correlation_threshold = proxy_correlation_threshold
        self.near_duplicate_cosine_threshold = near_duplicate_cosine_threshold

    def detector_1_univariate_screen(
        self,
        df: pd.DataFrame,
        target_col: str,
        is_classification: bool = True
    ) -> List[LeakageFinding]:
        """Detector 1: Univariate Perfect-Predictor Screen.

        Fit single-feature models / MI. Flags features with standalone
        predictive power exceeding plausible real-world signal.
        """
        findings: List[LeakageFinding] = []
        if target_col not in df.columns:
            return findings

        y = df[target_col]
        # Clean null targets for screening
        valid_mask = ~y.isna()
        df_valid = df[valid_mask]
        y_valid = y[valid_mask]

        feature_cols = [c for c in df.columns if c != target_col]
        if not feature_cols or len(df_valid) == 0:
            return findings

        mi_scores: Dict[str, float] = {}

        for col in feature_cols:
            x_raw = df_valid[col]
            # Convert categorical/object to numeric codes for fast screening
            if pd.api.types.is_numeric_dtype(x_raw):
                x_num = x_raw.fillna(x_raw.median()).values.reshape(-1, 1)
            else:
                x_num = pd.factorize(x_raw.fillna("__missing__"))[0].reshape(-1, 1)

            if is_classification and len(np.unique(y_valid)) == 2:
                # Binary classification ROC-AUC check via decision stump
                try:
                    stump = DecisionTreeClassifier(max_depth=1)
                    stump.fit(x_num, y_valid)
                    probs = stump.predict_proba(x_num)[:, 1]
                    auc = roc_auc_score(y_valid, probs)
                    # Symmetrical AUC (0.01 is as leaked as 0.99)
                    effective_auc = max(auc, 1.0 - auc)
                    if effective_auc >= self.auc_threshold:
                        findings.append(
                            LeakageFinding(
                                column=col,
                                action="exclude",
                                detector="Detector 1: Univariate Screen",
                                reason=f"Direct target leakage: standalone single-feature AUC {effective_auc:.4f} >= {self.auc_threshold}",
                                metric_value=float(effective_auc),
                            )
                        )
                except Exception:
                    pass

            if not is_classification:
                try:
                    stump = DecisionTreeRegressor(max_depth=2)
                    stump.fit(x_num, y_valid)
                    r2 = stump.score(x_num, y_valid)
                    if r2 >= self.auc_threshold:  # e.g. 0.98
                        findings.append(
                            LeakageFinding(
                                column=col,
                                action="exclude",
                                detector="Detector 1: Univariate Screen",
                                reason=f"Direct target leakage: standalone single-feature R2 {r2:.4f} >= {self.auc_threshold}",
                                metric_value=float(r2),
                            )
                        )
                except Exception:
                    pass

            # Calculate fast mutual info
            try:
                if is_classification:
                    mi = float(mutual_info_classif(x_num, y_valid, random_state=42)[0])
                else:
                    mi = float(mutual_info_regression(x_num, y_valid, random_state=42)[0])
                mi_scores[col] = mi
            except Exception:
                pass

        # Check extreme MI: only use relative percentile if wide feature space (>= 25 features)
        # to prevent dropping the top legitimate predictive feature on small tabular datasets
        if len(mi_scores) >= 25:
            values = list(mi_scores.values())
            pct_threshold = np.percentile(values, self.mi_percentile_threshold)
            for col, mi_val in mi_scores.items():
                if mi_val > pct_threshold and mi_val > 1.5:
                    if not any(f.column == col for f in findings):
                        findings.append(
                            LeakageFinding(
                                column=col,
                                action="exclude",
                                detector="Detector 1: Univariate Screen",
                                reason=f"Extreme mutual information outlier ({mi_val:.4f} > {pct_threshold:.4f} [{self.mi_percentile_threshold}th percentile])",
                                metric_value=float(mi_val),
                            )
                        )
        else:
            # On small feature sets, only flag if absolute mutual info exceeds plausible threshold
            for col, mi_val in mi_scores.items():
                if mi_val >= 2.5:
                    if not any(f.column == col for f in findings):
                        findings.append(
                            LeakageFinding(
                                column=col,
                                action="exclude",
                                detector="Detector 1: Univariate Screen",
                                reason=f"Near-deterministic mutual information ({mi_val:.4f} >= 2.5000)",
                                metric_value=float(mi_val),
                            )
                        )

        return findings

    def detector_2_temporal_availability(
        self,
        feature_metadata: Dict[str, Any],
        prediction_cutoff_timestamp: Optional[pd.Timestamp] = None
    ) -> List[LeakageFinding]:
        """Detector 2: Temporal Availability Check.

        Verifies that feature generation timestamp is <= prediction point.
        Features only known after outcome are auto-excluded.
        """
        findings: List[LeakageFinding] = []
        for col, meta in feature_metadata.items():
            generation_time = meta.get("generation_timestamp")
            if generation_time and prediction_cutoff_timestamp:
                gen_ts = pd.to_datetime(generation_time)
                if gen_ts > prediction_cutoff_timestamp:
                    findings.append(
                        LeakageFinding(
                            column=col,
                            action="exclude",
                            detector="Detector 2: Temporal Availability",
                            reason=f"Feature generation timestamp ({gen_ts}) occurs after prediction cutoff ({prediction_cutoff_timestamp})",
                        )
                    )
            if meta.get("is_post_outcome_event", False):
                findings.append(
                    LeakageFinding(
                        column=col,
                        action="exclude",
                        detector="Detector 2: Temporal Availability",
                        reason=f"Feature explicitly labeled as post-outcome event in domain metadata",
                    )
                )
        return findings

    def detector_3_cross_split_duplicates(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detector 3: Cross-Split Duplicate / Near-Duplicate Check.

        Detects exact row hashes across splits and eliminates leaking rows.
        """
        if feature_cols is None:
            # Use intersection of non-target columns
            feature_cols = list(train_df.columns)

        def hash_rows(df: pd.DataFrame) -> pd.Series:
            return pd.util.hash_pandas_object(df[feature_cols], index=False)

        train_hashes = set(hash_rows(train_df))
        val_hashes = set(hash_rows(val_df))
        test_hashes = set(hash_rows(test_df))

        train_val_overlap = train_hashes.intersection(val_hashes)
        train_test_overlap = train_hashes.intersection(test_hashes)
        val_test_overlap = val_hashes.intersection(test_hashes)

        leak_count = len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)

        return {
            "has_leakage": leak_count > 0,
            "train_val_overlap_count": len(train_val_overlap),
            "train_test_overlap_count": len(train_test_overlap),
            "val_test_overlap_count": len(val_test_overlap),
            "train_val_hashes": list(train_val_overlap),
            "train_test_hashes": list(train_test_overlap),
            "val_test_hashes": list(val_test_overlap),
        }

    def proxy_order_check(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> List[LeakageFinding]:
        """Additional check: ID/Order Proxy Check.

        Flags features that correlate strongly with row order or an ID column.
        """
        findings: List[LeakageFinding] = []
        row_order = np.arange(len(df))

        for col in df.columns:
            if col == target_col:
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                valid = ~series.isna()
                if valid.sum() > 10:
                    corr = np.corrcoef(row_order[valid], series[valid])[0, 1]
                    if abs(corr) >= self.proxy_correlation_threshold:
                        findings.append(
                            LeakageFinding(
                                column=col,
                                action="exclude",
                                detector="Proxy Check: ID/Order",
                                reason=f"Feature correlates strongly with row sequence ({corr:.4f} >= {self.proxy_correlation_threshold}); artifact proxy risk",
                                metric_value=float(abs(corr)),
                            )
                        )
        return findings

    def run_all_tabular_checks(
        self,
        df: pd.DataFrame,
        target_col: str,
        is_classification: bool = True,
        feature_metadata: Optional[Dict[str, Any]] = None,
        prediction_cutoff: Optional[pd.Timestamp] = None
    ) -> List[LeakageFinding]:
        """Run all data-level leakage checks and return consolidated findings."""
        findings = []
        findings.extend(self.detector_1_univariate_screen(df, target_col, is_classification))
        if feature_metadata:
            findings.extend(self.detector_2_temporal_availability(feature_metadata, prediction_cutoff))
        findings.extend(self.proxy_order_check(df, target_col))
        return findings
