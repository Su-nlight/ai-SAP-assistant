"""
Raw text extraction per supported file type.

Kept separate from `DocumentService` so adding a new file type (e.g. SAP
Notes in a bespoke export format) means adding one function + one registry
entry, not touching orchestration logic.
"""
from __future__ import annotations

import csv
import io

from app.core.exceptions import UnsupportedFileTypeError


def _extract_pdf(raw_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw_bytes))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx(raw_bytes: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(raw_bytes))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _extract_plain_text(raw_bytes: bytes) -> str:
    return raw_bytes.decode("utf-8", errors="ignore")


def _extract_csv(raw_bytes: bytes) -> str:
    text = raw_bytes.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    rows = [", ".join(row) for row in reader]
    return "\n".join(rows)


def _extract_html(raw_bytes: bytes) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_bytes.decode("utf-8", errors="ignore"), "html.parser")
    return soup.get_text(separator="\n")


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_plain_text,
    ".md": _extract_plain_text,
    ".csv": _extract_csv,
    ".html": _extract_html,
}


def extract_text(file_extension: str, raw_bytes: bytes) -> str:
    extractor = _EXTRACTORS.get(file_extension.lower())
    if extractor is None:
        raise UnsupportedFileTypeError(f"Unsupported file type '{file_extension}'.")
    return extractor(raw_bytes)
