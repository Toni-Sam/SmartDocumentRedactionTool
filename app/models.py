"""
app/models.py
-------------
Shared data shapes for the two-step detect-then-apply dispatcher flow.

DetectedSpan is the common unit produced by detect_document() and consumed
by apply_redactions(). It wraps whatever merger.py / docx_parser.py /
pdf_parser.py already return, adding:

  - id            : stable identifier, so a reviewer's decision on one span
                    (e.g. in a future Streamlit checklist) can be tracked
                    even if the manifest is reordered or partially edited.
  - include       : the reviewer's decision. True by default (everything
                    detected gets redacted unless a human unchecks it).
  - location      : format-specific position info.
                        - docx: None. docx_redactor.py matches spans by
                          exact text value across the whole document, not
                          by structural offset, so no location is needed
                          to apply the redaction. (Docx block-local start/
                          end from the detector aren't globally meaningful
                          anyway, since detection runs per-paragraph.)
                        - pdf: {"page": int, "start": int, "end": int} -
                          the page number and the character span *within
                          that page's reconstructed text*, exactly what
                          pdf_redactor.py needs to map back to word rects.

Manifest is the detect_document() output: the full set of spans for one
file, plus enough metadata to sanity-check that apply_redactions() is
being run against the right input later.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class DetectedSpan:
    id: str
    entity_type: str
    text: str
    source: str            # "presidio" | "local_context"
    score: float
    location: Optional[dict] = None
    include: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "DetectedSpan":
        return DetectedSpan(
            id=d["id"],
            entity_type=d["entity_type"],
            text=d["text"],
            source=d.get("source", "unknown"),
            score=d.get("score", 0.0),
            location=d.get("location"),
            include=d.get("include", True),
        )


@dataclass
class Manifest:
    input_path: str
    file_type: str          # "docx" | "pdf"
    created_at: str         # ISO 8601 timestamp
    spans: list = field(default_factory=list)   # list[DetectedSpan]

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "file_type": self.file_type,
            "created_at": self.created_at,
            "spans": [s.to_dict() for s in self.spans],
        }

    @staticmethod
    def from_dict(d: dict) -> "Manifest":
        return Manifest(
            input_path=d["input_path"],
            file_type=d["file_type"],
            created_at=d["created_at"],
            spans=[DetectedSpan.from_dict(s) for s in d.get("spans", [])],
        )

    def included_spans(self) -> list:
        """Spans a human has left flagged for redaction (include=True)."""
        return [s for s in self.spans if s.include]

    def summary(self) -> dict:
        """Quick counts for a reviewer-facing summary line."""
        by_type: dict = {}
        for s in self.spans:
            by_type[s.entity_type] = by_type.get(s.entity_type, 0) + 1
        return {
            "total_spans": len(self.spans),
            "included": len(self.included_spans()),
            "excluded": len(self.spans) - len(self.included_spans()),
            "by_entity_type": by_type,
        }


if __name__ == "__main__":
    # Quick round-trip sanity check
    span = DetectedSpan(
        id="test-1",
        entity_type="NG_NIN",
        text="12345678901",
        source="presidio",
        score=0.85,
        location={"page": 0, "start": 40, "end": 51},
    )
    manifest = Manifest(
        input_path="tests/NigerianSamples/DOCX_Redaction_Test.pdf",
        file_type="pdf",
        created_at="2026-07-22T00:00:00+00:00",
        spans=[span],
    )

    round_tripped = Manifest.from_dict(manifest.to_dict())
    assert round_tripped.spans[0].text == "12345678901"
    print("Round-trip OK:", round_tripped.summary())