"""Dataset PII and Protected Attribute Scanner (PRD v2 Section 9)."""

import pandas as pd
import re
from typing import List, Dict, Tuple, Any

class DatasetPIIScanner:
    """Scans structured datasets during ingestion for strict PII and protected attributes."""
    
    # Heuristics for strict PII (must drop/mask)
    STRICT_PII_PATTERNS = [
        r"email",
        r"ssn",
        r"social_security",
        r"phone",
        r"first_name",
        r"last_name",
        r"address",
        r"zip_code"
    ]
    
    # Heuristics for protected/sensitive attributes (keep, but flag for fairness)
    PROTECTED_ATTRIBUTES = [
        r"^age$",
        r"gender",
        r"sex",
        r"race",
        r"ethnicity",
        r"marital",
        r"religion",
        r"disability",
        r"pregnant"
    ]
    
    @classmethod
    def scan_dataframe(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Scan DataFrame for PII and protected attributes.
        Drops strict PII columns.
        Returns the sanitized DataFrame and a metadata dict detailing protected attributes.
        """
        cols_to_drop = []
        protected_cols = []
        
        for col in df.columns:
            col_lower = str(col).lower()
            
            # 1. Check strict PII header heuristics
            is_strict_pii = False
            for pat in cls.STRICT_PII_PATTERNS:
                if re.search(pat, col_lower):
                    is_strict_pii = True
                    break
                    
            # 1b. Check high-cardinality free-text (PII via notes/free text)
            if not is_strict_pii and (df[col].dtype == 'object' or df[col].dtype.name == 'string'):
                valid_series = df[col].dropna().astype(str)
                if len(valid_series) > 50:
                    unique_ratio = valid_series.nunique() / len(valid_series)
                    avg_len = valid_series.str.len().mean()
                    if unique_ratio > 0.90 and avg_len > 20:
                        is_strict_pii = True
                        
            if is_strict_pii:
                cols_to_drop.append(col)
                continue
                
            # 2. Check protected attributes
            is_protected = False
            for pat in cls.PROTECTED_ATTRIBUTES:
                if re.search(pat, col_lower):
                    is_protected = True
                    break
                    
            if is_protected:
                protected_cols.append(col)
                
        # Drop PII
        sanitized_df = df.drop(columns=cols_to_drop)
        
        metadata = {
            "dropped_pii_columns": cols_to_drop,
            "protected_attributes": protected_cols
        }
        
        return sanitized_df, metadata
