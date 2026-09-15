"""PII Screening and Data Redaction Engine (PRD v2 Section 9 & 60)."""

import re
from typing import Dict, Any, List, Set, Tuple, Optional
import pandas as pd


class PIIDetection:
    def __init__(self, column: str, pii_type: str, sample_match: str, confidence: float, recommended_action: str = "quarantine"):
        self.column = column
        self.pii_type = pii_type
        self.sample_match = sample_match
        self.confidence = confidence
        self.recommended_action = recommended_action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "pii_type": self.pii_type,
            "confidence": self.confidence,
            "recommended_action": self.recommended_action,
        }


class PIIScreeningEngine:
    """Detects Personally Identifiable Information (PII) before dataset reaches prompts, logs, or features."""

    PATTERNS = {
        "email": re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"),
        "phone": re.compile(r"^(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}$"),
        "ssn": re.compile(r"^\d{3}-\d{2}-\d{4}$"),
        "credit_card": re.compile(r"^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})$"),
        "ipv4": re.compile(r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"),
    }

    SUSPICIOUS_COLUMNS = {
        "ssn": "ssn",
        "social_security": "ssn",
        "email": "email",
        "email_address": "email",
        "phone": "phone",
        "phone_number": "phone",
        "mobile": "phone",
        "credit_card": "credit_card",
        "card_number": "credit_card",
        "ip_address": "ipv4",
        "password": "secret",
        "passport": "national_id",
        "full_name": "name",
        "first_name": "name",
        "last_name": "name",
    }

    @classmethod
    def screen_dataset(
        cls,
        df: pd.DataFrame,
        sample_size: int = 100,
        match_threshold: float = 0.10,
    ) -> List[PIIDetection]:
        """Screen dataset column names and data samples for PII."""
        detections: List[PIIDetection] = []
        sample_df = df.head(sample_size)

        for col in df.columns:
            clean_col = str(col).lower().replace(" ", "_").replace("-", "_")

            # 1. Header name heuristic
            for pattern_col, pii_type in cls.SUSPICIOUS_COLUMNS.items():
                if pattern_col in clean_col:
                    detections.append(
                        PIIDetection(
                            column=col,
                            pii_type=pii_type,
                            sample_match=f"Column header matched '{pattern_col}'",
                            confidence=0.95,
                            recommended_action="quarantine",
                        )
                    )
                    break

            # If already flagged by header, continue
            if any(d.column == col for d in detections):
                continue

            # 2. Content pattern sampling
            series = sample_df[col].dropna().astype(str)
            if len(series) == 0:
                continue

            for pii_type, regex in cls.PATTERNS.items():
                matches = series.apply(lambda val: bool(regex.match(val.strip())))
                match_ratio = matches.sum() / len(series)

                if match_ratio >= match_threshold:
                    sample_val = series[matches].iloc[0] if matches.any() else ""
                    # Redact sample for report safety
                    masked_sample = sample_val[:2] + "***" + sample_val[-2:] if len(sample_val) > 4 else "***"
                    detections.append(
                        PIIDetection(
                            column=col,
                            pii_type=pii_type,
                            sample_match=f"{matches.sum()}/{len(series)} values matched pattern ({masked_sample})",
                            confidence=min(1.0, match_ratio * 1.5),
                            recommended_action="quarantine",
                        )
                    )
                    break

        return detections

    @classmethod
    def sanitize_dataset(
        cls,
        df: pd.DataFrame,
        detections: List[PIIDetection],
        action: str = "drop",
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Sanitize dataset by dropping or masking quarantined PII columns."""
        sanitized = df.copy()
        flagged_cols = [d.column for d in detections]

        if action == "drop":
            sanitized = sanitized.drop(columns=flagged_cols, errors="ignore")
        elif action == "mask":
            for col in flagged_cols:
                if col in sanitized.columns:
                    sanitized[col] = "[REDACTED_PII]"

        return sanitized, flagged_cols
