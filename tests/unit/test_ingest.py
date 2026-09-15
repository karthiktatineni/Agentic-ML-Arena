"""Unit tests for Document Ingestor."""

import os
from app.knowledge.ingest import DocumentIngestor

def test_parse_text(tmp_path):
    text_file = tmp_path / "test.md"
    text_file.write_text("This is a markdown file.")
    
    content = DocumentIngestor.parse_file(str(text_file))
    assert "markdown file" in content

def test_unsupported_format(tmp_path):
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("dummy")
    
    import pytest
    with pytest.raises(ValueError, match="Unsupported file format"):
        DocumentIngestor.parse_file(str(bad_file))
