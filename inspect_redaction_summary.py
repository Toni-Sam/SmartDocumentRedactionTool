r"""
Diagnostic: runs the real detect -> apply flow on a test document and
prints exactly what apply_redactions() returns, so we know what fields
are actually available to surface in the GUI (e.g. counts of entities
redacted/skipped) instead of just dumping the whole dict via st.json().

Usage (from project root, with venv active):
    python inspect_redaction_summary.py tests\NigerianSamples\DOCX_Redaction_Test.docx
    python inspect_redaction_summary.py path\to\some_test.pdf
"""

import json
import sys
import tempfile
from pathlib import Path

from app.dispatcher import detect_document, apply_redactions, get_entities_for_review


def main(input_path: str) -> None:
    ext = Path(input_path).suffix.lower()
    out_path = str(Path(tempfile.gettempdir()) / f"summary_check_redacted{ext}")

    print(f"Detecting on: {input_path}\n")
    session = detect_document(input_path)

    entities = get_entities_for_review(session)
    print(f"Entities found: {len(entities)}")
    for e in entities:
        print(f"  id={e.get('id')} type={e.get('entity_type')} "
              f"text={e.get('text')!r} approved={e.get('approved')}")

    print(f"\nApplying redactions -> {out_path}\n")
    summary = apply_redactions(session, out_path)

    print("Returned summary object:")
    print(f"  type: {type(summary)}")

    if isinstance(summary, dict):
        print(f"  keys: {list(summary.keys())}\n")
        print("Full contents:")
        try:
            print(json.dumps(summary, indent=2, default=str))
        except TypeError:
            # Fall back if something inside isn't JSON-serializable
            for k, v in summary.items():
                print(f"  {k!r}: {v!r} ({type(v).__name__})")
    else:
        print(f"  (not a dict) repr: {summary!r}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_redaction_summary.py <path_to_docx_or_pdf>")
        sys.exit(1)
    main(sys.argv[1])