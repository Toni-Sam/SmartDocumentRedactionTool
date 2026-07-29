"""
dispatcher.py
--------------
Single entry point that routes a document to the right parser/detector,
and supports a two-step detect -> human review -> apply flow rather than
auto-redacting immediately.

WHY TWO STEPS
-------------
A PII redaction tool that silently redacts everything it detects has no
safety net for false positives/negatives - and a false negative here means
a real NIN or BVN leaks. The two-step flow lets a human (via the future
Streamlit GUI, or directly in code/tests) see exactly what was detected,
uncheck anything wrong, and only then commit to a destructive rewrite of
the document.

WHY DOCX AND PDF NEED DIFFERENT INTERNAL SHAPES
------------------------------------------------
- docx_redactor.redact_docx() re-opens the input file itself and matches
  entities by TEXT VALUE across the whole document. It only needs a flat
  list of {"text": ...} dicts - no parsed object needs to survive from
  detect -> apply.
- pdf_redactor.redact_pdf_from_detections() needs the actual
  (PDFPageBlock, entities) tuples pdf_parser.detect_pii_in_pdf() produces,
  because PDFPageBlock.word_spans carries fitz.Rect bounding boxes that
  only exist after parsing - they can't be reconstructed from a JSON
  manifest or from entity text alone.

So DetectionSession keeps a single flat, uniform `entities` list for
review (used by both file types + the GUI), plus a private `_page_results`
field that's populated ONLY for PDFs and holds the real PDFPageBlock
objects apply_redactions() needs. The entity dicts inside `_page_results`
are the SAME object references as the ones in `entities` - not copies -
so flipping `entity["approved"]` from a review UI is visible in both
places automatically, no separate sync step required.

GUI NOTE
--------
DetectionSession is meant to be held in memory across a request/session
(e.g. Streamlit's st.session_state), not serialized to JSON. session.entities
alone IS JSON-safe if you ever need to export/log a review manifest (it's
just text/ints/floats/bools) - it's only _page_results (PDFPageBlock /
fitz.Rect) that can't leave memory.

USAGE
-----
    from app.dispatcher import detect_document, apply_redactions, redact_document

    # Two-step flow (what the GUI will use):
    session = detect_document("client_form.docx")
    for e in session.entities:
        print(e["id"], e["entity_type"], e["text"], e["approved"])
    session.entities[3]["approved"] = False   # human unchecks one in review
    summary = apply_redactions(session, "client_form_redacted.docx")

    # One-shot convenience (everything approved, no review step):
    summary = redact_document("client_form.docx", "client_form_redacted.docx")
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from app.parsers.docx_parser import parse_and_detect
from app.parsers.pdf_parser import detect_pii_in_pdf, PDFPageBlock
from app.redactors.docx_redactor import redact_docx
from app.redactors.pdf_redactor import redact_pdf_from_detections


SUPPORTED_EXTENSIONS = {".docx": "docx", ".pdf": "pdf"}


# ---------------------------------------------------------------------------
# Session object
# ---------------------------------------------------------------------------

@dataclass
class DetectionSession:
    input_path: str
    file_type: str  # "docx" | "pdf"
    entities: list  # flat, review-ready list of entity dicts (see module docstring)
    _page_results: Optional[list] = field(default=None, repr=False)
    # PDF-only: list of (PDFPageBlock, entities) tuples with real fitz.Rect
    # data. None for docx, since redact_docx() re-parses the file itself.


def _next_id_counter():
    n = 0
    while True:
        yield n
        n += 1


# ---------------------------------------------------------------------------
# Step 1: detect
# ---------------------------------------------------------------------------

def detect_document(input_path: str) -> DetectionSession:
    """
    Parses + detects PII in `input_path`, routing by file extension, and
    returns a DetectionSession ready for human review. Nothing is redacted
    yet - call apply_redactions() (optionally after editing
    session.entities) to actually produce an output file.
    """
    ext = os.path.splitext(input_path)[1].lower()
    file_type = SUPPORTED_EXTENSIONS.get(ext)

    if file_type is None:
        raise ValueError(
            f"Unsupported file type '{ext}' for {input_path}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    ids = _next_id_counter()

    if file_type == "docx":
        raw_entities = parse_and_detect(input_path)
        entities = []
        for e in raw_entities:
            e = dict(e)  # own copy, don't mutate parser's return value
            e["id"] = next(ids)
            e["approved"] = True
            entities.append(e)

        return DetectionSession(
            input_path=input_path,
            file_type="docx",
            entities=entities,
            _page_results=None,
        )

    if file_type == "pdf":
        page_results = detect_pii_in_pdf(input_path)

        flat_entities = []
        for block, block_entities in page_results:
            for e in block_entities:
                # Mutate in place (not a copy) - session.entities and
                # _page_results must share the same dict objects so a
                # review-time edit to "approved" is visible to both.
                e["id"] = next(ids)
                e["approved"] = True
                e["page_num"] = block.page_num  # 0-indexed, matches PDFPageBlock
                flat_entities.append(e)

        return DetectionSession(
            input_path=input_path,
            file_type="pdf",
            entities=flat_entities,
            _page_results=page_results,
        )

    raise AssertionError(f"unreachable: unhandled file_type {file_type!r}")


# ---------------------------------------------------------------------------
# Review helpers (what the GUI will call between detect and apply)
# ---------------------------------------------------------------------------

def get_entities_for_review(session: DetectionSession) -> list:
    """
    Returns session.entities sorted for display: by page then position for
    PDFs, by detection order for docx (which has no page concept and is
    already deduplicated by text).
    """
    if session.file_type == "pdf":
        return sorted(session.entities, key=lambda e: (e.get("page_num", 0), e.get("start", 0)))
    return list(session.entities)


def set_approval(session: DetectionSession, entity_id: int, approved: bool) -> None:
    """Flip a single entity's approval status by id (e.g. a GUI checkbox toggle)."""
    for e in session.entities:
        if e["id"] == entity_id:
            e["approved"] = approved
            return
    raise KeyError(f"No entity with id={entity_id} in this session")


def set_all_approved(session: DetectionSession, approved: bool) -> None:
    """Bulk approve/reject-all - handy for a GUI 'select all' / 'clear all' control."""
    for e in session.entities:
        e["approved"] = approved


# ---------------------------------------------------------------------------
# Step 2: apply
# ---------------------------------------------------------------------------

def apply_redactions(session: DetectionSession, output_path: str, **redactor_kwargs) -> dict:
    """
    Redacts only the entities currently marked approved=True in
    session.entities, writing the result to output_path.

    redactor_kwargs are forwarded to the underlying redactor (e.g.
    redact_label="[REDACTED]" for docx, fill_color=(0,0,0) for pdf) so
    callers/GUI don't lose access to existing customization options.
    """
    if session.file_type == "docx":
        approved = [e for e in session.entities if e.get("approved", True)]
        return redact_docx(session.input_path, output_path, approved, **redactor_kwargs)

    if session.file_type == "pdf":
        filtered_page_results = [
            (block, [e for e in block_entities if e.get("approved", True)])
            for block, block_entities in session._page_results
        ]
        return redact_pdf_from_detections(
            session.input_path, output_path, filtered_page_results, **redactor_kwargs
        )

    raise AssertionError(f"unreachable: unhandled file_type {session.file_type!r}")


# ---------------------------------------------------------------------------
# One-shot convenience wrapper (no review step - everything approved)
# ---------------------------------------------------------------------------

def redact_document(input_path: str, output_path: str, **redactor_kwargs) -> dict:
    """
    Detect + apply in one call, with every detected entity approved.
    Useful for batch/CLI use and for tests/test_docx_pipeline.py /
    tests/test_pdf_pipeline.py, which don't need a review step. The GUI
    will use detect_document() + apply_redactions() directly instead, with
    a review step in between.
    """
    session = detect_document(input_path)
    return apply_redactions(session, output_path, **redactor_kwargs)


# ---------------------------------------------------------------------------
# Inline test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    input_path = sys.argv[1] if len(sys.argv) > 1 else "tests/NigerianSamples/DOCX_Redaction_Test.docx"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "tests/NigerianSamples/DOCX_Redaction_Test_redacted.docx"

    print(f"Step 1: detect_document({input_path})")
    session = detect_document(input_path)

    print(f"File type: {session.file_type}")
    print(f"Found {len(session.entities)} entities for review:\n")
    for e in get_entities_for_review(session):
        page_info = f" page={e['page_num']}" if session.file_type == "pdf" else ""
        print(f"  id={e['id']:<3} [{e['entity_type']}] {e['text']!r}"
              f" (score={e.get('score')}, source={e.get('source')}{page_info})")

    # Simulate a human review step: reject the first entity, if any, just
    # to demonstrate that apply_redactions() honors it.
    if session.entities:
        rejected_id = session.entities[0]["id"]
        set_approval(session, rejected_id, False)
        print(f"\n(simulated review: rejected entity id={rejected_id})")

    print(f"\nStep 2: apply_redactions() -> {output_path}")
    summary = apply_redactions(session, output_path)
    print(summary)