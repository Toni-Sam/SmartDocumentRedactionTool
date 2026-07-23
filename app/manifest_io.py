"""
app/manifest_io.py
-------------------
Reads and writes Manifest objects as JSON on disk.

This is the seam that makes the two-step flow actually two steps: it lets
detect_document() and apply_redactions() run in entirely separate process
invocations (or, later, separate Streamlit page loads), with a human
editing `include` flags on individual spans in between.

Usage:
    from app.dispatcher import detect_document
    from app.manifest_io import save_manifest, load_manifest

    manifest = detect_document("input.docx")
    save_manifest(manifest, "input.manifest.json")

    # ... human reviews input.manifest.json, flips some "include": false ...

    manifest = load_manifest("input.manifest.json")
    apply_redactions("input.docx", manifest, "input_redacted.docx")
"""

import json

from app.models import Manifest


def save_manifest(manifest: Manifest, path: str) -> None:
    """Writes a Manifest to disk as human-editable JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)


def load_manifest(path: str) -> Manifest:
    """Reads a Manifest back from a JSON file previously written by save_manifest()."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Manifest.from_dict(data)


if __name__ == "__main__":
    import sys
    import os
    from app.models import DetectedSpan

    test_path = sys.argv[1] if len(sys.argv) > 1 else "tests/NigerianSamples/manifest_io_test.json"

    manifest = Manifest(
        input_path="tests/NigerianSamples/DOCX_Redaction_Test.docx",
        file_type="docx",
        created_at="2026-07-22T00:00:00+00:00",
        spans=[
            DetectedSpan(
                id="span-1",
                entity_type="NG_NIN",
                text="12345678901",
                source="presidio",
                score=0.85,
            ),
            DetectedSpan(
                id="span-2",
                entity_type="PERSON_NAME",
                text="Oladele Peter",
                source="local_context",
                score=0.7,
                include=False,  # simulate a human excluding a false positive
            ),
        ],
    )

    save_manifest(manifest, test_path)
    print(f"Saved manifest to {test_path}")

    loaded = load_manifest(test_path)
    print("Loaded back:", loaded.summary())

    os.remove(test_path)
    print("Cleaned up test file.")