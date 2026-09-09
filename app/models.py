"""
app/models.py
-------------
DetectedSpan: a formal, serializable shape for one detected entity.

STATUS: not currently used by dispatcher.py. dispatcher.py builds its own
plain dicts directly (with "page_num", "page_type", "id", "approved" keys)
rather than constructing DetectedSpan objects - that's the live code path
today. This class is kept as a target shape for later, in case a feature
needs entities in a formal, JSON-safe, versioned form (e.g. an audit
report or a persisted review manifest) rather than dispatcher's in-memory
dicts. Until something actually builds/consumes DetectedSpan objects,
treat this file as a spec, not a dependency.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class DetectedSpan:
    id: str
    entity_type: str
    text: str
    source: str             # "presidio" | "local_context" | "masakhaner"
    score: float
    location: Optional[dict] = None
    include: bool = True

    # OCR-sourced spans only (None for text-layer PDFs and docx):
    page_type: Optional[str] = None   # "text" | "scanned"
    ocr_confidence: Optional[float] = None  # 0-100, from WordSpan.confidence

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
            page_type=d.get("page_type"),
            ocr_confidence=d.get("ocr_confidence"),
        )


if __name__ == "__main__":
    span = DetectedSpan(
        id="test-1",
        entity_type="NG_NIN",
        text="12345678901",
        source="presidio",
        score=0.85,
        location={"page": 0, "start": 40, "end": 51},
    )
    round_tripped = DetectedSpan.from_dict(span.to_dict())
    assert round_tripped.text == "12345678901"
    print("Round-trip OK:", round_tripped)