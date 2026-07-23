"""
docx_redactor.py
----------------
Replaces detected PII spans with a redaction label (default "[REDACTED]")
in .docx files, using python-docx.

THE CORE PROBLEM
----------------
Word doesn't store a paragraph's visible text as one continuous string. It
splits it across multiple <w:r> "run" elements based on formatting changes,
spell-check boundaries, autocorrect, and other internal reasons unrelated to
the text's actual content. So an entity like "Emeka Okonkwo" might physically
live in the XML as:

    Run 1: "Emeka Ok"
    Run 2: "onk"
    Run 3: "wo"

A naive `run.text.replace(...)` only catches an entity when it happens to sit
entirely inside a single run. This module reconstructs each paragraph's full
text, finds entity spans at the character level (across run boundaries), and
rewrites the paragraph so the redacted text replaces the original.

Trade-off: when a redaction spans multiple runs, this collapses the whole
paragraph down to a single run carrying the new text (using the first run's
formatting). That guarantees the redaction itself is correct, at the cost of
losing run-level formatting variety (e.g. only part of a paragraph being
bold) in paragraphs that get redacted. Paragraphs with no matches are left
completely untouched.

WHAT GETS SCANNED
-----------------
- Body paragraphs
- Table cells, including nested tables
- Headers and footers, including first-page/even-page variants, across all
  sections

Known limitation: text inside text boxes / floating shapes is not covered
(python-docx doesn't expose these through the normal paragraph API; they'd
need raw OOXML traversal via document.element as a follow-up).

USAGE
-----
    from app.redactors.docx_redactor import redact_docx

    entities = [
        {"text": "Emeka Okonkwo", "entity_type": "PERSON_NAME"},
        {"text": "08012345678",   "entity_type": "NG_PHONE"},
    ]
    result = redact_docx("input.docx", "output_redacted.docx", entities)
    print(result)
    # {"entities_matched": 2, "paragraphs_redacted": 1, "output": "output_redacted.docx"}

Entities just need a "text" key - this is designed to slot directly into the
output of docx_parser.parse_and_detect(), which already deduplicates by text.
"""

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


# ---------------------------------------------------------------------------
# Paragraph-level cross-run text replacement
# ---------------------------------------------------------------------------

def _paragraph_full_text(paragraph: Paragraph) -> str:
    """Full visible text of a paragraph, concatenated across all its runs."""
    return "".join(run.text for run in paragraph.runs)


def _find_spans_for_entities(full_text: str, entities: list[dict]) -> list[tuple[int, int]]:
    """
    Given a paragraph's full text and the entity list (matched by text
    value), return sorted, non-overlapping [start, end) spans to redact.

    Entities are matched longest-text-first, so "Emeka Okonkwo" claims its
    span before "Emeka" gets a chance to - preventing a first name from
    being redacted on its own while the surname next to it is missed.
    """
    spans: list[tuple[int, int]] = []

    for entity in sorted(entities, key=lambda e: len(e.get("text", "")), reverse=True):
        needle = entity.get("text", "")
        if not needle:
            continue

        search_from = 0
        while True:
            idx = full_text.find(needle, search_from)
            if idx == -1:
                break
            candidate = (idx, idx + len(needle))
            overlaps_existing = any(candidate[0] < s[1] and s[0] < candidate[1] for s in spans)
            if not overlaps_existing:
                spans.append(candidate)
            search_from = idx + len(needle)

    spans.sort()
    return spans


def _redact_paragraph(paragraph: Paragraph, spans: list[tuple[int, int]], label: str) -> bool:
    """
    Replace the given [start, end) spans (positions in the paragraph's
    full text) with `label`. Spans must already be sorted and
    non-overlapping. Returns True if the paragraph text actually changed.
    """
    if not spans or not paragraph.runs:
        return False

    full_text = _paragraph_full_text(paragraph)
    if not full_text:
        return False

    pieces = []
    cursor = 0
    for start, end in spans:
        if start < cursor or start >= len(full_text):
            continue  # stale/invalid span, skip defensively
        end = min(end, len(full_text))
        pieces.append(full_text[cursor:start])
        pieces.append(label)
        cursor = end
    pieces.append(full_text[cursor:])
    new_text = "".join(pieces)

    if new_text == full_text:
        return False

    # Collapse to a single run so the new text renders correctly, keeping
    # the first run's formatting (font, bold, etc.) as the paragraph's style.
    first_run = paragraph.runs[0]
    first_run.text = new_text
    for run in paragraph.runs[1:]:
        run.text = ""

    return True


def _redact_paragraph_against_entities(paragraph: Paragraph, entities: list[dict], label: str) -> bool:
    full_text = _paragraph_full_text(paragraph)
    if not full_text:
        return False
    spans = _find_spans_for_entities(full_text, entities)
    if not spans:
        return False
    return _redact_paragraph(paragraph, spans, label)


# ---------------------------------------------------------------------------
# Structural walkers: body, tables (incl. nested), headers/footers
# ---------------------------------------------------------------------------

def _iter_table_paragraphs(table: Table):
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                yield paragraph
            for nested_table in cell.tables:
                yield from _iter_table_paragraphs(nested_table)


def _iter_body_and_table_paragraphs(document: Document):
    for paragraph in document.paragraphs:
        yield paragraph
    for table in document.tables:
        yield from _iter_table_paragraphs(table)


def _iter_header_footer_paragraphs(document: Document):
    for section in document.sections:
        parts = (
            section.header,
            section.footer,
            section.first_page_header,
            section.first_page_footer,
            section.even_page_header,
            section.even_page_footer,
        )
        for part in parts:
            if part is None:
                continue
            for paragraph in part.paragraphs:
                yield paragraph
            for table in part.tables:
                yield from _iter_table_paragraphs(table)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def redact_docx(input_path: str, output_path: str, entities: list[dict],
                 redact_label: str = "[REDACTED]") -> dict:
    """
    Redact every occurrence of `entities` (each a dict with at least a
    "text" key) throughout a .docx file's body, tables, and headers/footers,
    saving the result to `output_path`.

    Returns a summary dict: {"entities_matched", "paragraphs_redacted", "output"}
    """
    document = Document(input_path)

    usable_entities = [e for e in entities if e.get("text")]

    paragraphs_redacted = 0
    matched_texts = set()

    all_paragraphs = (
        list(_iter_body_and_table_paragraphs(document))
        + list(_iter_header_footer_paragraphs(document))
    )

    for paragraph in all_paragraphs:
        full_text_before = _paragraph_full_text(paragraph)
        if not full_text_before:
            continue

        changed = _redact_paragraph_against_entities(paragraph, usable_entities, redact_label)
        if changed:
            paragraphs_redacted += 1
            for entity in usable_entities:
                if entity["text"] in full_text_before:
                    matched_texts.add(entity["text"])

    document.save(output_path)

    return {
        "entities_matched": len(matched_texts),
        "paragraphs_redacted": paragraphs_redacted,
        "output": output_path,
    }


# ---------------------------------------------------------------------------
# Inline test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sample_entities = [
        {"text": "Emeka Okonkwo", "entity_type": "PERSON_NAME"},
        {"text": "08012345678", "entity_type": "NG_PHONE"},
        {"text": "Ikeja, Lagos", "entity_type": "ADDRESS"},
    ]

    input_file = sys.argv[1] if len(sys.argv) > 1 else "sample_input.docx"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "sample_redacted.docx"

    print(f"Redacting {input_file} -> {output_file}")
    result = redact_docx(input_file, output_file, sample_entities)
    print(result)