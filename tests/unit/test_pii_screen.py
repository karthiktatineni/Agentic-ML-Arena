"""Unit tests for PII screening and dataset sanitization (PRD v2 Section 9 & 60)."""

import pytest
import pandas as pd
from app.governance.pii_screen import PIIScreeningEngine


def test_pii_screening_header_detection():
    """Verify that columns with suspicious PII header names are quarantined."""
    df = pd.DataFrame({
        "customer_id": [101, 102, 103],
        "user_email": ["a@b.com", "c@d.com", "e@f.com"],
        "social_security_num": ["000-00-0000", "111-11-1111", "222-22-2222"],
        "annual_income": [50000, 60000, 75000],
    })

    detections = PIIScreeningEngine.screen_dataset(df)
    flagged_cols = [d.column for d in detections]

    assert "user_email" in flagged_cols
    assert "social_security_num" in flagged_cols
    assert "annual_income" not in flagged_cols


def test_pii_screening_content_detection():
    """Verify that unnamed columns containing PII content are caught by pattern regex."""
    df = pd.DataFrame({
        "field_alpha": ["alice@work.org", "bob@domain.io", "charlie@company.com"],
        "field_beta": ["192.168.1.1", "10.0.0.1", "172.16.0.5"],
        "field_gamma": ["safe_cat_1", "safe_cat_2", "safe_cat_3"],
    })

    detections = PIIScreeningEngine.screen_dataset(df)
    flagged_types = {d.column: d.pii_type for d in detections}

    assert flagged_types.get("field_alpha") == "email"
    assert flagged_types.get("field_beta") == "ipv4"
    assert "field_gamma" not in flagged_types


def test_pii_sanitization_drop_and_mask():
    """Verify sanitize_dataset correctly drops or masks flagged PII columns."""
    df = pd.DataFrame({
        "email": ["user1@test.com", "user2@test.com"],
        "score": [85, 92],
    })

    detections = PIIScreeningEngine.screen_dataset(df)

    # Test drop
    df_dropped, dropped_cols = PIIScreeningEngine.sanitize_dataset(df, detections, action="drop")
    assert "email" not in df_dropped.columns
    assert "score" in df_dropped.columns
    assert dropped_cols == ["email"]

    # Test mask
    df_masked, masked_cols = PIIScreeningEngine.sanitize_dataset(df, detections, action="mask")
    assert "email" in df_masked.columns
    assert df_masked["email"].iloc[0] == "[REDACTED_PII]"
