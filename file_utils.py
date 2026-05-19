"""
file_utils.py — Extract text from uploaded files (PDF, DOCX, images).
"""

import os
import tempfile


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extract text content from a file based on its extension."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return _extract_pdf(file_bytes)
    elif ext in (".doc", ".docx"):
        return _extract_docx(file_bytes)
    elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        return _extract_image(file_bytes, ext)
    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        return ""


def _extract_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts).strip()
    except Exception as e:
        return f"[Error reading PDF: {e}]"


def _extract_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file."""
    try:
        from docx import Document
        import io

        doc = Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        return f"[Error reading DOCX: {e}]"


def _extract_image(file_bytes: bytes, ext: str) -> str:
    """Extract text from an image using OCR (pytesseract)."""
    try:
        from PIL import Image
        import pytesseract
        import io

        image = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(image)
        return text.strip() if text.strip() else "[No text detected in image]"
    except ImportError:
        return "[OCR not available — install pytesseract and Tesseract-OCR]"
    except Exception as e:
        return f"[Error reading image: {e}]"
