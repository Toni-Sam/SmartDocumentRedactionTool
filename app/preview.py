"""
preview.py
----------
Read-only rendering for the GUI's document preview: lets a reviewer see
detected PII highlighted in place (color-coded by status) without having
to download the redacted output first. This is aimed at the "manual
redaction" pain point - some PII (e.g. a one-off registration number) has
no pattern recognizer, so a reviewer has to spot it visually before
typing it into dispatcher.add_manual_entity().

This module never modifies the underlying document. PDF/image pages are
rendered to an in-memory image (fitz page pixmap -> PIL Image), with
highlight rectangles drawn onto the PIL image itself - nothing is drawn
into the source file's own content stream or annotations, and the
throwaway fitz.Document opened here is never saved.

fitz.open() transparently handles standalone raster images (JPG/PNG/
TIFF/BMP) as an implicit one-page document, so every function here works
unchanged for session.file_type in ("pdf", "image").

COLOR CODING (shared between the PDF/image and DOCX previews)
---------------------------------------------------------------
- Approved entity   -> translucent coral fill   (WILL be redacted)
- Rejected entity   -> translucent gray fill, dashed outline (PDF only;
                       DOCX has no per-pixel outline to draw) (will NOT
                       be redacted)
- Manually added    -> translucent purple fill (source == "manual")

For DOCX, a given text value can be matched by more than one entity dict
(e.g. an auto-detected entity and a later manual one sharing the same
text). Since redact_docx() redacts document-wide by text value, that text
WILL be redacted if ANY matching entity is approved - so DOCX span color
priority is: approved (any) > manual (any, all unapproved) > rejected.
PDF entities are per-occurrence, not per-text-value, so no such grouping
is needed there - an entity's own approved/source fields decide its color
directly.

All highlights are semi-transparent so the underlying text stays legible
underneath - the point of this preview is to help a reviewer SPOT missed
PII, not to simulate the final redacted output.
"""

from itertools import count as _count
from typing import List

import fitz
from PIL import Image, ImageDraw
from docx import Document

from app.redactors.pdf_redactor import _rects_for_span
from app.redactors.docx_redactor import (
    _iter_body_and_table_paragraphs,
    _iter_header_footer_paragraphs,
    _paragraph_full_text,
    _find_spans_for_entities,
)

# RGBA overlay colors for the PIL-drawn PDF/image preview (0-255 per
# channel; alpha controls translucency so text stays readable underneath).
_COLOR_APPROVED = (231, 76, 60, 90)      # coral
_COLOR_REJECTED = (149, 165, 166, 60)    # gray
_COLOR_MANUAL = (155, 89, 182, 100)      # purple
_OUTLINE_REJECTED = (127, 140, 141, 200)

# CSS rgba() equivalents for the DOCX HTML preview.
HTML_COLOR_APPROVED = "rgba(231, 76, 60, 0.35)"
HTML_COLOR_REJECTED = "rgba(149, 165, 166, 0.30)"
HTML_COLOR_MANUAL = "rgba(155, 89, 182, 0.40)"


# ---------------------------------------------------------------------------
# PDF / image: page image with highlight overlays
# ---------------------------------------------------------------------------

def get_page_count(input_path: str) -> int:
    """Number of renderable pages - always 1 for a standalone image."""
    doc = fitz.open(input_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def _entity_pixel_color(entity: dict):
    if entity.get("source") == "manual":
        return _COLOR_MANUAL
    return _COLOR_APPROVED if entity.get("approved") else _COLOR_REJECTED


def render_page_with_highlights(
    input_path: str,
    page_results: list,
    page_num: int,
    zoom: float = 2.0,
) -> Image.Image:
    """
    Renders page `page_num` (0-indexed) of the PDF/image at `input_path`
    as a PIL Image, with a translucent highlight box drawn over every
    entity on that page - color-coded per the module docstring. This
    opens its own throwaway fitz.Document purely for rendering; nothing
    is written back to the source file.

    `page_results` is the same (PDFPageBlock, entities) list dispatcher.py
    already tracks as session._page_results - this function only reads
    from it.
    """
    doc = fitz.open(input_path)
    try:
        page = doc[page_num]
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("RGBA")
    finally:
        doc.close()

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for block, entities in page_results:
        if block.page_num != page_num:
            continue
        for entity in entities:
            rects = _rects_for_span(block, entity["start"], entity["end"])
            for rect in rects:
                scaled = (rect.x0 * zoom, rect.y0 * zoom, rect.x1 * zoom, rect.y1 * zoom)
                draw.rectangle(scaled, fill=_entity_pixel_color(entity))
                if entity.get("source") != "manual" and not entity.get("approved"):
                    draw.rectangle(scaled, outline=_OUTLINE_REJECTED, width=2)

    return Image.alpha_composite(image, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# DOCX: highlighted HTML preview
# ---------------------------------------------------------------------------

def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _docx_span_color(text_value: str, entities_by_text: dict) -> str:
    candidates = entities_by_text.get(text_value, [])
    if any(c.get("approved") for c in candidates):
        return HTML_COLOR_APPROVED
    if any(c.get("source") == "manual" for c in candidates):
        return HTML_COLOR_MANUAL
    return HTML_COLOR_REJECTED


def render_docx_preview_html(input_path: str, entities: List[dict]) -> str:
    """
    Renders the whole DOCX (body, tables, headers/footers - same scan
    order as docx_redactor.py) as one HTML block, wrapping every detected
    entity's text in a color-coded <mark>. Uses the SAME longest-match-
    first, non-overlapping span logic docx_redactor.py's own
    _find_spans_for_entities() uses, so what's highlighted here always
    matches what redact_docx() will actually redact.

    Returns an HTML string meant for st.markdown(..., unsafe_allow_html=True)
    inside a scrollable container - it does not include a <html>/<body>
    wrapper, just a sequence of <p> blocks.
    """
    document = Document(input_path)
    counter = _count(1)
    all_paragraphs = (
        list(_iter_body_and_table_paragraphs(document, counter))
        + list(_iter_header_footer_paragraphs(document, counter))
    )

    usable_entities = [e for e in entities if e.get("text")]
    entities_by_text: dict = {}
    for e in usable_entities:
        entities_by_text.setdefault(e["text"], []).append(e)

    html_parts = []
    for paragraph, _ in all_paragraphs:
        full_text = _paragraph_full_text(paragraph)
        if not full_text.strip():
            continue

        spans = _find_spans_for_entities(full_text, usable_entities)
        if not spans:
            html_parts.append(f"<p>{_escape_html(full_text)}</p>")
            continue

        pieces = []
        cursor = 0
        for start, end, text_value in spans:
            pieces.append(_escape_html(full_text[cursor:start]))
            color = _docx_span_color(text_value, entities_by_text)
            pieces.append(
                f'<mark style="background-color:{color}; padding:1px 2px; '
                f'border-radius:3px;">{_escape_html(full_text[start:end])}</mark>'
            )
            cursor = end
        pieces.append(_escape_html(full_text[cursor:]))
        html_parts.append(f"<p>{''.join(pieces)}</p>")

    return "".join(html_parts) if html_parts else "<p><em>(No extractable text found.)</em></p>"


def render_legend_html() -> str:
    """Shared color-key strip, used above both the PDF and DOCX previews."""
    chip = (
        '<span style="background-color:{color}; padding:2px 8px; '
        'border-radius:4px; margin-right:10px; font-size:0.82em;">{label}</span>'
    )
    return (
        '<div style="margin-bottom:8px;">'
        + chip.format(color=HTML_COLOR_APPROVED, label="will be redacted")
        + chip.format(color=HTML_COLOR_REJECTED, label="rejected")
        + chip.format(color=HTML_COLOR_MANUAL, label="manually added")
        + "</div>"
    )


if __name__ == "__main__":
    # Run as: python -m app.preview tests\NigerianSamples\sample.pdf
    import sys

    test_path = sys.argv[1] if len(sys.argv) > 1 else "tests/NigerianSamples/DOCX_Redaction_Test.pdf"
    from app.parsers.pdf_parser import detect_pii_in_pdf

    print(f"Rendering preview for: {test_path}")
    page_results = detect_pii_in_pdf(test_path)
    print(f"{get_page_count(test_path)} page(s), {sum(len(e) for _, e in page_results)} entities total")

    img = render_page_with_highlights(test_path, page_results, page_num=0)
    out_path = "preview_page_1.png"
    img.save(out_path)
    print(f"Saved preview of page 1 to {out_path}")