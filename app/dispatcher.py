"""
app/dispatcher.py
------------------
Single entry point for the Smart Document Redaction Tool's two-step flow.

    manifest = detect_document(input_path)
    save_manifest(manifest, "manifest.json")          # (app.manifest_io)

    # --- human review happens here: edit manifest.json, flip include ---

    manifest = load_manifest("manifest.json")          # (app.manifest_io)
    apply_redactions(input_path, manifest, output_path)

detect_document() routes to docx_parser.parse_and_detect() or
pdf_parser.detect_pii_in_pdf() depending on file extension, and wraps
whatever they return into DetectedSpan objects (app.models) - a shared
shape that's the same regardless of file type, so a future GUI review
screen doesn't need format-specific logic.

apply_redactions() routes to docx_redactor.redact_docx() or
pdf_redactor.redact_pdf_from_detections(), filtering to only the spans a
human has left with include=True.

WHY PDF NEEDS A REBUILD STEP AND DOCX DOESN'T
-----------------------------------------------
docx_redactor.redact_docx() matches entities by exact text value across
the whole document - it doesn't need positional info at apply time. So a
docx manifest's spans carry no `location`, and applying is a straight
pass-through of {"text", "entity_type"} pairs.

pdf_redactor needs word-level bounding boxes (fitz.Rect objects) to redact
the content stream, and those aren't JSON-serializable - they can't live
in a saved manifest. So for PDFs, detect_document() stores only
{"page", "start", "end"} in each span's location, and apply_redactions()
re-parses the PDF fresh (parse_pdf()) to regenerate those rects, then
matches the manifest's spans back onto the freshly parsed pages by page
number. This is safe as long as the same input file is used for both
steps - PyMuPDF's word extraction is deterministic for a given file.
"""

import os
import uuid
from datetime import datetime, timezone

from app.models import DetectedSpan, Manifest
from app.parsers.docx_parser import parse_and_detect as parse_and_detect_docx
from app.parsers.pdf_parser import detect_pii_in_pdf, parse_pdf, PDFPageBlock
from app.redactors.docx_redactor import redact_docx
from app.redactors.pdf_redactor import redact_pdf_from_detections

SUPPORTED_EXTENSIONS = {".docx", ".pdf"}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _detect_docx(input_path: str) -> list:
    """Runs the existing docx detection pipeline and wraps results as DetectedSpans."""
    entities = parse_and_detect_docx(input_path)
    spans = []
    for e in entities:
        spans.append(DetectedSpan(
            id=str(uuid.uuid4()),
            entity_type=e.get("entity_type", "UNKNOWN"),
            text=e.get("text", ""),
            source=e.get("source", "unknown"),
            score=e.get("score", 0.0),
            location=None,  # docx_redactor matches by text, not position
            include=True,
        ))
    return spans


def _detect_pdf(input_path: str) -> list:
    """Runs the existing PDF detection pipeline and wraps results as DetectedSpans."""
    page_results = detect_pii_in_pdf(input_path)
    spans = []
    for block, entities in page_results:
        for e in entities:
            spans.append(DetectedSpan(
                id=str(uuid.uuid4()),
                entity_type=e.get("entity_type", "UNKNOWN"),
                text=e.get("text", ""),
                source=e.get("source", "unknown"),
                score=e.get("score", 0.0),
                location={
                    "page": block.page_num,
                    "start": e["start"],
                    "end": e["end"],
                },
                include=True,
            ))
    return spans


def detect_document(input_path: str) -> Manifest:
    """
    Runs the appropriate detection pipeline for `input_path` based on its
    extension and returns a Manifest ready to be saved for human review.
    """
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".docx":
        spans = _detect_docx(input_path)
        file_type = "docx"
    elif ext == ".pdf":
        spans = _detect_pdf(input_path)
        file_type = "pdf"
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    return Manifest(
        input_path=input_path,
        file_type=file_type,
        created_at=datetime.now(timezone.utc).isoformat(),
        spans=spans,
    )


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------

def _rebuild_pdf_page_results(input_path: str, manifest: Manifest):
    """
    Re-parses `input_path` to regenerate PDFPageBlocks (with their rects),
    then matches the manifest's included spans back onto those blocks by
    page number, reconstructing the (PDFPageBlock, entities) shape
    pdf_redactor.redact_pdf_from_detections() expects.
    """
    blocks = parse_pdf(input_path)
    blocks_by_page = {b.page_num: b for b in blocks}

    page_entities: dict = {}
    for span in manifest.included_spans():
        if not span.location or "page" not in span.location:
            continue
        page_num = span.location["page"]
        page_entities.setdefault(page_num, []).append({
            "start": span.location["start"],
            "end": span.location["end"],
            "text": span.text,
            "entity_type": span.entity_type,
        })

    page_results = []
    for page_num, entities in page_entities.items():
        block = blocks_by_page.get(page_num)
        if block is not None:
            page_results.append((block, entities))
    return page_results


def apply_redactions(input_path: str, manifest: Manifest, output_path: str) -> dict:
    """
    Applies redactions for every span in `manifest` with include=True,
    routing to the docx or pdf redactor based on manifest.file_type.

    Returns whatever summary dict the underlying redactor produces.
    """
    if manifest.input_path != input_path:
        print(
            f"Warning: manifest was built from '{manifest.input_path}', "
            f"but is being applied against '{input_path}'. Proceeding, but "
            f"double-check these refer to the same document."
        )

    included = manifest.included_spans()

    if manifest.file_type == "docx":
        entities = [{"text": s.text, "entity_type": s.entity_type} for s in included]
        return redact_docx(input_path, output_path, entities)

    elif manifest.file_type == "pdf":
        page_results = _rebuild_pdf_page_results(input_path, manifest)
        return redact_pdf_from_detections(input_path, output_path, page_results)

    else:
        raise ValueError(f"Unsupported file type in manifest: '{manifest.file_type}'")


# ---------------------------------------------------------------------------
# Inline test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run as:
    #   python -m app.dispatcher tests\NigerianSamples\DOCX_Redaction_Test.docx
    import sys
    from app.manifest_io import save_manifest, load_manifest

    input_path = sys.argv[1] if len(sys.argv) > 1 else "tests/NigerianSamples/DOCX_Redaction_Test.docx"
    ext = os.path.splitext(input_path)[1].lower()
    manifest_path = input_path.rsplit(".", 1)[0] + ".manifest.json"
    output_path = input_path.rsplit(".", 1)[0] + "_redacted" + ext

    print(f"[1/3] Detecting PII in {input_path}...")
    manifest = detect_document(input_path)
    print(f"      {manifest.summary()}")

    print(f"[2/3] Saving manifest to {manifest_path} for review...")
    save_manifest(manifest, manifest_path)
    print("      (In a real run, a human edits 'include' flags here.)")

    print(f"[3/3] Loading manifest and applying redactions -> {output_path}...")
    manifest = load_manifest(manifest_path)
    result = apply_redactions(input_path, manifest, output_path)
    print(f"      {result}")