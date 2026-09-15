"""Unit tests for Text PII Scanner."""

from app.knowledge.pii_screen import TextPIIScanner

def test_text_pii_scanner():
    raw_text = "Contact me at user@example.com or call (555) 123-4567. IP is 192.168.1.1."
    
    scrubbed = TextPIIScanner.scrub_text(raw_text)
    
    assert "user@example.com" not in scrubbed
    assert "[EMAIL_REDACTED]" in scrubbed
    assert "(555) 123-4567" not in scrubbed
    assert "[PHONE_REDACTED]" in scrubbed
    assert "192.168.1.1" not in scrubbed
    assert "[IP_ADDRESS_REDACTED]" in scrubbed
