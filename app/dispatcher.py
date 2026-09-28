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

PAGE-LEVEL REVIEW FLAGS
------------------------
flagged_for_ocr_review (set in pdf_parser.py's detect_pii_in_pdf()) is a
PAGE-level signal - "this page classified as text yielded almost nothing,
it might actually be an unrecognized scan" - not an entity-level one. It
lives on PDFPageBlock, which only survives in the private _page_results
list. Without pulling it out, a GUI has no way to show a reviewer "page 4
might have been missed" - the flag would be invisible outside this module.

flagged_pages surfaces that: a flat list of page_nums (0-indexed, matching
PDFPageBlock.page_num) where flagged_for_ocr_review is True, built once at
detect time. Deliberately kept separate from `entities` rather than
stamped onto every entity dict for that page - the flag describes the
page as a whole, not any individual entity, so a page-level banner in the
GUI is the correct UX, not a per-row annotation. Empty list for docx and
for PDFs where nothing was flagged.

GUI NOTE
--------
DetectionSession is meant to be held in memory across a request/session
(e.g. Streamlit's st.session_state), not serialized to JSON. session.entities
alone IS JSON-safe if you ever need to export/log a review manifest (it's
just text/ints/floats/bools) - it's only _page_results (PDFPageBlock /
fitz.Rect) that can't leave memory. flagged_pages is also JSON-safe (a
plain list of ints).

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

    # Redaction-policy flow (set default approvals by entity type, then
    # let the human review/override as usual):
    from app.redaction_policies import resolve_policy_types
    allowed = resolve_policy_types("Financial IDs only")
    set_approval_by_types(session, allowed)

    # Manual mark: human supplies a value the pipeline missed (e.g. a
    # unique registration number with no pattern recognizer). Matches
    # case-insensitively, document-wide, and adds one pre-approved entity
    # per occurrence found (see add_manual_entity()'s own docstring):
    added = add_manual_entity(session, "REG-2024-8841", "MANUAL_PII")
    if not added:
        print("No matches found for that value.")
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from app.parsers.docx_parser import parse_and_detect
from app.parsers.pdf_parser import detect_pii_in_pdf, PDFPageBlock
from app.redactors.docx_redactor import redact_docx
from app.redactors.pdf_redactor import redact_pdf_from_detections


SUPPORTED_EXTENSIONS = {
    ".docx": "docx",
    ".pdf": "pdf",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".tiff": "image",
    ".tif": "image",
    ".bmp": "image",
}


# ---------------------------------------------------------------------------
# Session object
# ---------------------------------------------------------------------------

@dataclass
class DetectionSession:
    input_path: str
    file_type: str  # "docx" | "pdf" | "image"
    entities: list  # flat, review-ready list of entity dicts (see module docstring)
    _page_results: Optional[list] = field(default=None, repr=False)
    # PDF-only: list of (PDFPageBlock, entities) tuples with real fitz.Rect
    # data. None for docx, since redact_docx() re-parses the file itself.
    flagged_pages: list = field(default_factory=list)
    # Page numbers (0-indexed) where PDFPageBlock.flagged_for_ocr_review is
    # True - a page classified "text" that yielded thin content and zero
    # entities, possibly an unrecognized scan. Page-level, not entity-level
    # (see module docstring). Always empty for docx.


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
            flagged_pages=[],
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
                e["page_type"] = block.page_type  # "text" | "scanned"
                flat_entities.append(e)

        # Page-level review signal, pulled out of PDFPageBlock since
        # _page_results is private plumbing the GUI shouldn't reach into
        # directly (see module docstring: PAGE-LEVEL REVIEW FLAGS).
        flagged_pages = [
            block.page_num for block, _ in page_results
            if block.flagged_for_ocr_review
        ]

        return DetectionSession(
            input_path=input_path,
            file_type="pdf",
            entities=flat_entities,
            _page_results=page_results,
            flagged_pages=flagged_pages,
        )

    if file_type == "image":
        # Standalone image file (.jpg/.png/.tiff/.bmp) - not a PDF, so there's
        # no page loop or PDFPageBlock.page_num concept. Routed straight to
        # scanned_pdf_parser.py's image-input path.
        try:
            from app.parsers.scanned_pdf_parser import extract_image_block
        except ImportError as exc:
            raise NotImplementedError(
                f"{input_path!r} is an image file, but "
                f"app/parsers/scanned_pdf_parser.py doesn't exist yet - OCR "
                f"support hasn't been built. This file cannot be parsed until it is."
            ) from exc

        block, block_entities = extract_image_block(input_path, dpi=300)
        entities = []
        for e in block_entities:
            e["id"] = next(ids)
            e["approved"] = True
            e["page_num"] = 0
            e["page_type"] = "scanned"
            entities.append(e)

        # Standalone images have no PyMuPDF page-classification step (no
        # "text" vs "scanned" routing decision to second-guess - it's
        # always scanned), so flagged_for_ocr_review doesn't apply here.
        return DetectionSession(
            input_path=input_path,
            file_type="image",
            entities=entities,
            _page_results=[(block, entities)],
            flagged_pages=[],
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


def set_approval_by_types(session: DetectionSession, allowed_types: set) -> None:
    """
    Sets approved=True for every entity whose entity_type is in
    allowed_types, and approved=False for everything else.

    This is the mechanism a redaction policy (see app/redaction_policies.py)
    uses to set default approvals for a chosen policy - e.g. "Financial
    IDs only" resolves to a set of entity_type strings, and this function
    applies that set as a bulk starting point. It doesn't detect anything
    new and doesn't touch entities' other fields - a human reviewer can
    still flip any individual entity afterward via set_approval(), exactly
    as with set_all_approved(). Calling this again with a different
    allowed_types (e.g. the user switches policy) simply re-applies from
    scratch; it does not merge with whatever approvals were set before.
    """
    for e in session.entities:
        e["approved"] = e.get("entity_type", "UNKNOWN") in allowed_types


# ---------------------------------------------------------------------------
# Manual entity addition (human marks PII the pipeline missed)
# ---------------------------------------------------------------------------
# Covers the "mark" half of manual mark/unmark - "unmark" already exists via
# set_approval()/set_all_approved() above, since a manually added entity is
# just another dict in session.entities with "approved" flippable exactly
# like an auto-detected one.
#
# Matching here is DELIBERATELY different from the automatic pipeline:
# case-insensitive, partial (substring) search, document-wide, redacting
# EVERY occurrence found - agreed to cover cases like a unique registration
# number with no pattern recognizer, where the human already knows the
# exact value and just wants it found and redacted everywhere it appears.
#
# - DOCX: mirrors how redact_docx() already matches - by exact TEXT VALUE,
#   document-wide. Since that matching is case-sensitive, case-insensitivity
#   is handled here at ADD time instead: each paragraph is searched
#   case-insensitively, and one entity is created per DISTINCT actual
#   casing found in the document (so "ABC123" and "abc123" both get their
#   own entity and both get redacted, even though redact_docx() itself
#   never does case-insensitive matching). Reuses docx_redactor.py's own
#   paragraph walkers so "found here" and "found by the redactor later"
#   can never disagree.
# - PDF/image: mirrors how these are already tracked - per OCCURRENCE, via
#   start/end offsets into each PDFPageBlock's `text`. Offsets are computed
#   from the SAME `block.text` pdf_parser.py already built (and word_spans
#   were built against), so pdf_redactor.py's existing _rects_for_span()
#   lookup resolves them correctly with no changes to pdf_redactor.py.

def add_manual_entity(session: DetectionSession, query: str, entity_type: str) -> list:
    """
    Adds a human-supplied PII value to `session` after detect_document()
    has already run, so it flows through review/redaction exactly like an
    auto-detected entity (appears pre-approved in session.entities /
    get_entities_for_review(), can be unchecked via set_approval() same as
    any other row, and is picked up by apply_redactions() unchanged).

    `query` is matched case-insensitively as a substring, document-wide -
    see the section docstring above for why this differs from the
    automatic pipeline's exact-span matching, and how DOCX vs PDF/image
    differ in what gets created.

    Returns the list of newly created entity dicts. An empty list means
    `query` wasn't found anywhere in the document - callers (the GUI)
    should surface a "no matches found" message and let the human retry
    with a different value, per the agreed behavior; nothing is raised for
    this expected case.
    """
    query = query.strip()
    if not query:
        return []

    pattern = re.compile(re.escape(query), re.IGNORECASE)
    next_id = max((e.get("id", -1) for e in session.entities), default=-1) + 1
    new_entities = []

    if session.file_type == "docx":
        from docx import Document
        from itertools import count as _count
        from app.redactors.docx_redactor import (
            _iter_body_and_table_paragraphs,
            _iter_header_footer_paragraphs,
            _paragraph_full_text,
        )

        document = Document(session.input_path)
        counter = _count(1)
        all_paragraphs = (
            list(_iter_body_and_table_paragraphs(document, counter))
            + list(_iter_header_footer_paragraphs(document, counter))
        )

        found_texts = set()
        for paragraph, _ in all_paragraphs:
            full_text = _paragraph_full_text(paragraph)
            if not full_text:
                continue
            for m in pattern.finditer(full_text):
                found_texts.add(full_text[m.start():m.end()])

        already_manual = {
            e["text"] for e in session.entities
            if e.get("source") == "manual" and e.get("entity_type") == entity_type
        }

        for text_value in found_texts:
            if text_value in already_manual:
                continue  # already added in a prior call with the same type
            entity = {
                "text": text_value,
                "entity_type": entity_type,
                "score": 1.0,
                "source": "manual",
                "id": next_id,
                "approved": True,
            }
            next_id += 1
            session.entities.append(entity)
            new_entities.append(entity)

        return new_entities

    if session.file_type in ("pdf", "image"):
        for block, block_entities in session._page_results:
            for m in pattern.finditer(block.text):
                start, end = m.start(), m.end()

                already_added = any(
                    e.get("source") == "manual"
                    and e.get("start") == start
                    and e.get("end") == end
                    and e.get("entity_type") == entity_type
                    for e in block_entities
                )
                if already_added:
                    continue  # already added in a prior call, same span+type

                entity = {
                    "text": block.text[start:end],
                    "entity_type": entity_type,
                    "start": start,
                    "end": end,
                    "score": 1.0,
                    "source": "manual",
                    "id": next_id,
                    "approved": True,
                    "page_num": block.page_num,
                    "page_type": block.page_type,
                }
                next_id += 1

                # For PDF, block_entities is a distinct list from
                # session.entities (aggregated from many blocks) - append
                # to both. For image, block_entities IS session.entities
                # (same list object, single block) - the identity check
                # below skips the second append so it isn't added twice.
                block_entities.append(entity)
                if not any(existing is entity for existing in session.entities):
                    session.entities.append(entity)
                new_entities.append(entity)

        return new_entities

    raise NotImplementedError(
        f"Manual entity addition isn't supported for file_type={session.file_type!r} yet."
    )


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

    if session.file_type == "image":
        try:
            from app.redactors.scanned_pdf_redactor import redact_image_from_detections
        except ImportError as exc:
            raise NotImplementedError(
                f"{session.input_path!r} is an image file, but "
                f"app/redactors/scanned_pdf_redactor.py doesn't exist yet - "
                f"OCR redaction hasn't been built. Cannot apply redactions until it is."
            ) from exc

        block, block_entities = session._page_results[0]
        approved = [e for e in block_entities if e.get("approved", True)]
        return redact_image_from_detections(
            session.input_path, output_path, block, approved, **redactor_kwargs
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

    if session.flagged_pages:
        print(f"⚠ Pages flagged for OCR review (thin text, zero entities - "
              f"possible missed scan): {session.flagged_pages}")

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

    # Demonstrate a redaction policy: restrict to financial IDs only, then
    # show that non-financial entities are no longer approved.
    from app.redaction_policies import resolve_policy_types
    financial_only = resolve_policy_types("Financial IDs only")
    set_approval_by_types(session, financial_only)
    approved_count = sum(1 for e in session.entities if e["approved"])
    print(
        f"\n(simulated policy: 'Financial IDs only' -> "
        f"{approved_count}/{len(session.entities)} entities now approved)"
    )
    # Restore full approval before the actual apply step below, so this
    # inline test's final output matches its historical behavior (only
    # the single manually-rejected entity above stays un-redacted).
    for e in session.entities:
        e["approved"] = True
    set_approval(session, rejected_id, False)

    # Demonstrate manual entity addition: a value the pipeline wouldn't
    # have flagged on its own (no matching pattern recognizer), added by
    # a human reviewer and matched case-insensitively/document-wide.
    print("\n--- Manual entity addition check ---")
    manual_query = "REG-2024-8841"
    added = add_manual_entity(session, manual_query, "MANUAL_PII")
    if added:
        print(f"Added {len(added)} manual entity(ies) for {manual_query!r}:")
        for e in added:
            page_info = f" page={e['page_num']}" if "page_num" in e else ""
            print(f"  id={e['id']:<3} [{e['entity_type']}] {e['text']!r}{page_info}")
    else:
        print(f"No matches found for {manual_query!r} (expected unless the "
              f"test document happens to contain that exact value).")

    print(f"\nStep 2: apply_redactions() -> {output_path}")
    summary = apply_redactions(session, output_path)
    print(summary)