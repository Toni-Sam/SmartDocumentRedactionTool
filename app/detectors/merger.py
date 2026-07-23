"""
merger.py

Combines detection results from all layers of the pipeline:
- presidio_detector.py (Presidio defaults + Nigerian regex patterns)
- local_context_detector.py (gazetteer + heuristic context detection:
  names, LGAs, states of origin - replaces the old Claude API layer,
  no external calls, no cost)

Resolves overlapping spans between detectors and returns a single
deduplicated, sorted list of entities ready for the redactors.
"""

from app.detectors.presidio_detector import detect_with_presidio
from app.detectors.local_context_detector import detect_local_context


# ---------------------------------------------------------------------------
# Priority rules
# ---------------------------------------------------------------------------
# When two detectors flag overlapping spans, this decides which one wins.
# local_context_detector is preferred for free-text entities (names,
# places) since Presidio/spaCy's default English NER isn't tuned for
# Nigerian names. Presidio/regex is preferred for fixed-format entities
# (IDs, numbers) because it's deterministic and pattern-based there too -
# for those types both detectors are reliable, so this mostly matters
# when Presidio's generic PERSON tag disagrees with our gazetteer LGA/
# state matches on the same span.

CONTEXT_PREFERRED_TYPES = {
    "PERSON_NAME",
    "LGA",
    "STATE_OF_ORIGIN",
    "ETHNIC_GROUP",
    "ADDRESS",
}

PATTERN_PREFERRED_TYPES = {
    "NG_NIN",
    "NG_BVN",
    "NG_PASSPORT",
    "NG_PHONE",
    "NG_PLATE",
    "EMAIL_ADDRESS",
    "ACCOUNT_NUMBER",
    "NG_VIN",
    "NG_DRIVERS_LICENSE",
    "NG_TIN",
    "NG_NUBAN",
    "NG_RSA_PIN",
    "NG_NYSC",
    "NG_NYSC_STATE_CODE",
    "NG_CAC_RC",
    "NG_NHIS",
}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_local_context_results(context_results: list[dict]) -> list[dict]:
    """
    local_context_detector.py returns: {"text", "type", "start", "end", "score"}
    This converts it to the common shape used across all detectors:
    {"text", "entity_type", "start", "end", "score", "source"}

    Unlike the old Claude layer (which had no real confidence signal and
    was hardcoded to 0.9), local_context_detector assigns a genuine
    per-type score - gazetteer matches (LGA/state/ethnicity) score high
    since they're exact matches against a known list; name/address
    heuristics score lower since they're pattern-based guesses. That
    score is preserved here rather than overwritten.
    """
    normalized = []
    for r in context_results:
        normalized.append({
            "text": r.get("text", ""),
            "entity_type": r.get("type", "UNKNOWN"),
            "start": r.get("start"),
            "end": r.get("end"),
            "score": r.get("score", 0.5),
            "source": "local_context",
        })
    return normalized


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------

def _spans_overlap(a: dict, b: dict) -> bool:
    """True if two entity spans share any character position."""
    if a["start"] is None or b["start"] is None:
        return False
    return a["start"] < b["end"] and b["start"] < a["end"]


def _pick_winner(a: dict, b: dict) -> dict:
    """
    Given two overlapping detections, decide which one to keep.

    Priority order:
    1. Entity-type preference (local_context wins for names/places, patterns win for IDs)
    2. Higher confidence score
    3. Longer span (more specific match)
    """
    a_type, b_type = a["entity_type"], b["entity_type"]

    if a["source"] == "local_context" and a_type in CONTEXT_PREFERRED_TYPES:
        return a
    if b["source"] == "local_context" and b_type in CONTEXT_PREFERRED_TYPES:
        return b

    if a["source"] == "presidio" and a_type in PATTERN_PREFERRED_TYPES:
        return a
    if b["source"] == "presidio" and b_type in PATTERN_PREFERRED_TYPES:
        return b

    if a["score"] != b["score"]:
        return a if a["score"] > b["score"] else b

    a_len = a["end"] - a["start"]
    b_len = b["end"] - b["start"]
    return a if a_len >= b_len else b


def _merge_overlaps(detections: list[dict]) -> list[dict]:
    """
    Sorts detections by start position, then walks through resolving
    any overlapping spans down to a single winner each.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: (d["start"], d["end"]))

    merged = [detections[0]]

    for current in detections[1:]:
        last = merged[-1]

        if _spans_overlap(last, current):
            winner = _pick_winner(last, current)
            merged[-1] = winner
        else:
            merged.append(current)

    return merged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_all(text: str, language: str = "en") -> list[dict]:
    """
    Runs the full detection pipeline (Presidio + Nigerian patterns +
    local gazetteer/heuristic context detection) and returns a single
    merged, deduplicated list of entities sorted by position in the text.

    No external API calls anywhere in this pipeline.

    Output shape (ready for redactors):
        {
            "text": "...",
            "entity_type": "...",
            "start": N,
            "end": N,
            "score": 0.0-1.0,
            "source": "presidio" | "local_context",
        }
    """
    if not text or not text.strip():
        return []

    presidio_results = detect_with_presidio(text, language=language)
    context_raw = detect_local_context(text)
    context_results = _normalize_local_context_results(context_raw)

    all_detections = presidio_results + context_results
    merged = _merge_overlaps(all_detections)

    return merged


# ---------------------------------------------------------------------------
# Inline test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_text = (
        "Applicant: Chukwuemeka Okonkwo, NIN: 12345678901, "
        "BVN: 22334455667, Phone: 08012345678, "
        "Passport: A01234567, Plate: LAG 123AB. "
        "He resides in Nnewi North, Anambra State, and is of Igbo ethnicity."
    )

    print(f"Testing merger.py against sample text:\n{test_text}\n")

    results = detect_all(test_text)

    if not results:
        print("No entities detected — check that both detectors are working.")
    else:
        for r in results:
            print(
                f"[{r['source']}] {r['entity_type']}: '{r['text']}' "
                f"(pos {r['start']}-{r['end']}, score {r['score']})"
            )