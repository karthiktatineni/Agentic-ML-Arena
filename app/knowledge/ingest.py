"""Document Ingestion (PRD v2 Section 7)."""

import os


class DocumentIngestor:
    """Parses text from various document formats."""
    
    @classmethod
    def parse_file(cls, filepath: str) -> str:
        """Parse text from a file based on its extension."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == ".pdf":
            return cls._parse_pdf(filepath)
        elif ext == ".docx":
            return cls._parse_docx(filepath)
        elif ext in (".md", ".txt"):
            return cls._parse_text(filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
    @staticmethod
    def _parse_pdf(filepath: str) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("pypdf is required for PDF ingestion. Install with: pip install pypdf") from exc
        text = ""
        try:
            reader = PdfReader(filepath)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF: {str(e)}")
        return text
        
    @staticmethod
    def _parse_docx(filepath: str) -> str:
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError(
                "python-docx is required for DOCX ingestion. Install with: pip install python-docx"
            ) from exc
        text = ""
        try:
            doc = Document(filepath)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            raise RuntimeError(f"Failed to parse DOCX: {str(e)}")
        return text
        
    @staticmethod
    def _parse_text(filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
