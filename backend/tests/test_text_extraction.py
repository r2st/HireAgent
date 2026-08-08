"""Resume text extraction: format detection, per-format extractors, failures."""

from __future__ import annotations

import pytest

from app.services.text_extraction import (
    MIN_USEFUL_CHARS,
    ExtractionError,
    clean_text,
    detect_format,
    extract_text,
)
from tests.factories import SAMPLE_RESUME, make_docx, make_pdf

DOCX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)


class TestDetectFormat:
    @pytest.mark.parametrize(
        ("content_type", "filename", "expected"),
        [
            ("application/pdf", None, "pdf"),
            (DOCX_CONTENT_TYPE, None, "docx"),
            ("text/plain", None, "txt"),
            # Charset parameters must not defeat the lookup.
            ("text/plain; charset=utf-8", None, "txt"),
            ("APPLICATION/PDF", None, "pdf"),
        ],
    )
    def test_content_type_wins(self, content_type, filename, expected):
        assert detect_format(content_type, filename) == expected

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("cv.pdf", "pdf"),
            ("Resume.DOCX", "docx"),
            ("notes.txt", "txt"),
            ("readme.md", "txt"),
        ],
    )
    def test_falls_back_to_extension(self, filename, expected):
        # Browsers frequently send octet-stream for uploads.
        assert detect_format("application/octet-stream", filename) == expected

    def test_rejects_unknown_format(self):
        with pytest.raises(ExtractionError) as exc:
            detect_format("image/png", "scan.png")
        assert exc.value.details["filename"] == "scan.png"

    def test_rejects_when_nothing_identifies_the_file(self):
        with pytest.raises(ExtractionError):
            detect_format(None, None)


class TestCleanText:
    def test_collapses_horizontal_whitespace_but_keeps_lines(self):
        cleaned = clean_text("Ada   Lovelace\n\n\n\nStaff  Engineer")
        # Line structure separates roles and dates, so it must survive.
        assert cleaned == "Ada Lovelace\n\nStaff Engineer"

    def test_normalises_line_endings(self):
        assert clean_text("a\r\nb\rc") == "a\nb\nc"

    def test_expands_ligatures_from_pdf_extraction(self):
        assert clean_text("ofﬁce workﬂow") == "office workflow"

    def test_strips_control_characters(self):
        assert clean_text("Ada\x00\x07 Lovelace") == "Ada Lovelace"


class TestExtractText:
    def test_extracts_plain_text(self):
        text, fmt = extract_text(
            SAMPLE_RESUME.encode(), content_type="text/plain", filename="cv.txt"
        )
        assert fmt == "txt"
        assert "ada.lovelace@example.com" in text

    def test_extracts_pdf(self):
        text, fmt = extract_text(
            make_pdf(SAMPLE_RESUME), content_type="application/pdf", filename="cv.pdf"
        )
        assert fmt == "pdf"
        assert "Ada Lovelace" in text
        assert "Staff Engineer at Analytical Engines" in text

    def test_extracts_docx(self):
        text, fmt = extract_text(
            make_docx(SAMPLE_RESUME),
            content_type=DOCX_CONTENT_TYPE,
            filename="cv.docx",
        )
        assert fmt == "docx"
        assert "ada.lovelace@example.com" in text

    def test_extracts_docx_tables(self):
        # Many resumes lay experience out in a table, which paragraph-only
        # extraction would silently drop.
        data = make_docx(
            SAMPLE_RESUME,
            table_rows=[["Staff Engineer", "Analytical Engines", "2021-Present"]],
        )
        text, _ = extract_text(data, content_type=DOCX_CONTENT_TYPE, filename="cv.docx")
        assert "Staff Engineer | Analytical Engines | 2021-Present" in text

    def test_decodes_utf16_text(self):
        text, _ = extract_text(
            SAMPLE_RESUME.encode("utf-16"), content_type="text/plain", filename="cv.txt"
        )
        assert "Ada Lovelace" in text

    def test_rejects_empty_file(self):
        with pytest.raises(ExtractionError, match="empty"):
            extract_text(b"", content_type="text/plain", filename="cv.txt")

    def test_rejects_unreadable_pdf(self):
        with pytest.raises(ExtractionError, match="Could not read the PDF"):
            extract_text(
                b"not a pdf at all", content_type="application/pdf", filename="cv.pdf"
            )

    def test_rejects_unreadable_docx(self):
        with pytest.raises(ExtractionError, match="Could not read the DOCX"):
            extract_text(
                b"PK\x03\x04 garbage",
                content_type=DOCX_CONTENT_TYPE,
                filename="cv.docx",
            )

    def test_reports_scanned_image_as_needing_ocr(self):
        # A PDF with almost no extractable text is the signature of a scan.
        short = "x" * (MIN_USEFUL_CHARS // 4)
        with pytest.raises(ExtractionError, match="OCR") as exc:
            extract_text(
                make_pdf(short), content_type="application/pdf", filename="scan.pdf"
            )
        assert exc.value.details["format"] == "pdf"
