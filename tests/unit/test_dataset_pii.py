"""Unit tests for Dataset PII Scanner."""

import pandas as pd
from app.governance.dataset_pii import DatasetPIIScanner

def test_dataset_pii_scanner():
    df = pd.DataFrame({
        "customer_id": [1, 2],
        "email_address": ["a@b.com", "c@d.com"],
        "user_age": [25, 30],
        "biological_sex": ["M", "F"],
        "income": [50000, 60000]
    })
    
    sanitized_df, metadata = DatasetPIIScanner.scan_dataframe(df)
    
    # Check strict PII is dropped
    assert "email_address" not in sanitized_df.columns
    assert "email_address" in metadata["dropped_pii_columns"]
    
    # Check protected attributes are kept but flagged
    assert "user_age" not in metadata["protected_attributes"] # regex is ^age$
    assert "biological_sex" in metadata["protected_attributes"]
    assert "biological_sex" in sanitized_df.columns
    
    # Check normal columns are kept
    assert "customer_id" in sanitized_df.columns
    assert "income" in sanitized_df.columns
