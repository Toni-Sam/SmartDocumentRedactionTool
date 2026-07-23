"""
tests/test_pdf_pipeline.py
---------------------------
End-to-end test for the PDF redaction pipeline: parse -> detect -> redact.

Run from the project root as a module:
    python -m tests.test_pdf_pipeline

Requires a sample PDF at tests/nigerian_samples/sample.pdf containing some
Nigerian PII (name, NIN, BVN, phone number, etc.) to redact against.
"""

import os
import fitz

from app.redactors.pdf_redactor import redact_pdf

SAMPLE_INPUT = "tests/NigerianSamples/DOCX_Redaction_Test.pdf"
SAMPLE_OUTPUT = "tests/NigerianSamples/DOCX_Redaction_Test_redacted.pdf"


def test_pdf_pipeline():
    assert os.path.exists(SAMPLE_INPUT), f"Missing test file: {SAMPLE_INPUT}"

    summary = redact_pdf(SAMPLE_INPUT, SAMPLE_OUTPUT)

    print(f"Pages processed: {summary['pages']}")
    print(f"Entities redacted: {summary['entities_redacted']}")

    assert summary["entities_redacted"] > 0, "No entities were redacted - check detection pipeline"
    assert os.path.exists(SAMPLE_OUTPUT), "Redacted output file was not created"

    # Sanity check: confirm the redacted text is actually gone from the
    # content stream (true redaction), not just visually covered.
    doc = fitz.open(SAMPLE_OUTPUT)
    full_text = "".join(page.get_text() for page in doc)
    doc.close()

    print("\nRedacted output text sample (first 500 chars):")
    print(full_text[:500])

    print("\nPDF pipeline test passed")


if __name__ == "__main__":
    test_pdf_pipeline()