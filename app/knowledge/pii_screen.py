"""Document Text PII Screening (PRD v2 Section 9 - Supplemental)."""

import re

class TextPIIScanner:
    """Regex-based scrubber to remove sensitive patterns from free text before embedding."""
    
    # Common PII regex patterns
    PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "PHONE": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
    }
    
    @classmethod
    def scrub_text(cls, text: str) -> str:
        """Replace PII patterns with placeholders."""
        if not text:
            return text
            
        scrubbed = text
        for label, pattern in cls.PATTERNS.items():
            scrubbed = re.sub(pattern, f"[{label}_REDACTED]", scrubbed)
            
        return scrubbed
