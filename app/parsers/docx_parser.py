"""
docx_parser.py
--------------
Extracts text from a .docx file - body paragraphs, tables (including nested
tables), and headers/footers - and runs it through the detection pipeline
(presidio_detector + local_context_detector, combined via merger.py) to
produce a list of PII entities ready for docx_redactor.py.

WHY THIS EXISTS
---------------
Your detectors work on plain text. A .docx file's visible text is scattered
across many separate XML elements (paragraph runs, table cells nested
arbitrarily deep, header/footer parts per section). This module gathers all
of that text, runs it through detection one block at a time (so a name near
a form label like "Name:" stays next to that label for local_context_detector
to anchor on), and returns a single deduplicated entity list.

NOTE ON THE MERGER IMPORT
--------------------------
This assumes merger.py exposes a function shaped like:

    merge_detections(text: str) -> list[dict]

...that internally calls both detect_with_presidio() and
detect_local_context(), resolves overlaps, and returns entities shaped like:

    {"text": ..., "entity_type": ..., "start": ..., "end": ..., "score": ..., "source": ...}

If your merger.py's top-level function has a different name, just change
the import line below to match it.

Usage:
    from app.parsers.docx_parser import parse_and_detect

    entities = parse_and_detect("client_form.docx")
    # -> [{"text": "Chukwuemeka Okonkwo", "entity_type": "PERSON_NAME", ...}, ...]

    from app.redactors.docx_redactor import redact_docx
    redact_docx("client_form.docx", "client_form_redacted.docx", entities)
"""

from docx import Document
from docx.table import Table

from app.detectors.merger import detect_all  # <-- adjust name if needed


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _iter_table_paragraph_texts(table: Table):
    """Yield non-empty paragraph text from every cell in a table, recursing
    into nested tables (Nigerian government/bank forms often nest tables
    inside table cells for layout)."""
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                text = paragraph.text
                if text.strip():
                    yield text
            for nested_table in cell.tables:
                yield from _iter_table_paragraph_texts(nested_table)


def _iter_header_footer_texts(document: Document):
    """Yield non-empty paragraph text from every header/footer variant
    (default, first-page, even-page) across all sections."""
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
                if paragraph.text.strip():
                    yield paragraph.text
            for table in part.tables:
                yield from _iter_table_paragraph_texts(table)


def extract_text_blocks(docx_path: str) -> dict:
    """
    Extract text from a .docx file, grouped by source, as lists of
    non-empty text blocks (one block per paragraph, or per table-cell
    paragraph). Detection runs per-block rather than on one giant string,
    since Nigerian names/LGAs/field labels are almost always resolvable
    within a single paragraph or cell, and per-block offsets stay
    meaningful (no need to re-map positions back across block boundaries).

    Returns:
        {
            "body": [str, ...],             # top-level body paragraphs
            "tables": [str, ...],           # paragraphs inside table cells
            "headers_footers": [str, ...],  # paragraphs inside headers/footers
        }

    Known limitation: text inside text boxes / floating shapes is not
    extracted here (python-docx doesn't expose these directly; they'd
    need raw OOXML traversal via document.element). Flag if your sample
    documents use text boxes and this can be added as a follow-up.
    """
    document = Document(docx_path)

    body_blocks = [p.text for p in document.paragraphs if p.text.strip()]

    table_blocks = []
    for table in document.tables:
        table_blocks.extend(_iter_table_paragraph_texts(table))

    header_footer_blocks = list(_iter_header_footer_texts(document))

    return {
        "body": body_blocks,
        "tables": table_blocks,
        "headers_footers": header_footer_blocks,
    }


def extract_full_text(docx_path: str) -> str:
    """Convenience: every extracted block joined into one newline-separated
    string. Useful for a quick eyeball check, not used by parse_and_detect
    (which detects block-by-block instead)."""
    blocks = extract_text_blocks(docx_path)
    all_blocks = blocks["body"] + blocks["tables"] + blocks["headers_footers"]
    return "\n".join(all_blocks)


# ---------------------------------------------------------------------------
# Detection orchestration
# ---------------------------------------------------------------------------

def parse_and_detect(docx_path: str) -> list[dict]:
    """
    Extract text from every source in the .docx file, run each block
    through the merged detection pipeline, and return a single
    deduplicated list of entities (deduplicated by exact text value)
    ready to hand to docx_redactor.redact_docx().

    Deduplication is intentional and matches how docx_redactor.py works:
    it redacts by matching text value across the whole document, so a
    name detected once in the body and again identically in a table
    only needs to appear once in the entities list.
    """
    blocks = extract_text_blocks(docx_path)
    all_blocks = blocks["body"] + blocks["tables"] + blocks["headers_footers"]

    seen_texts = set()
    entities = []

    for block in all_blocks:
        for entity in detect_all(block):
            text = entity.get("text", "")
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)
            entities.append(entity)

    return entities


# ---------------------------------------------------------------------------
# Inline test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    docx_path = sys.argv[1] if len(sys.argv) > 1 else "sample_input.docx"

    print(f"Extracting text blocks from {docx_path}...")
    blocks = extract_text_blocks(docx_path)
    print(f"  body paragraphs: {len(blocks['body'])}")
    print(f"  table cell paragraphs: {len(blocks['tables'])}")
    print(f"  header/footer paragraphs: {len(blocks['headers_footers'])}")

    print("\nRunning detection pipeline (presidio + local_context via merger)...")
    entities = parse_and_detect(docx_path)

    print(f"\nFound {len(entities)} unique entities:")
    for e in entities:
        print(f"  [{e.get('entity_type')}] {e.get('text')!r} "
              f"(score={e.get('score')}, source={e.get('source')})")