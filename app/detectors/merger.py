"""
merger.py

Combines detection results from all layers of the pipeline:
- presidio_detector.py (Presidio defaults + Nigerian regex patterns)
- local_context_detector.py (gazetteer + heuristic context detection:
  names, LGAs, states of origin - replaces the old Claude API layer,
  no external calls, no cost)
- masakhaner_detector.py (mBERT model fine-tuned on MasakhaNER - PERSON
  entities only, using learned Hausa/Igbo/Yoruba name patterns instead
  of a curated fragment list, so it generalizes to names
  local_context_detector's NIGERIAN_NAME_FRAGMENTS list would miss)

Resolves overlapping spans between detectors and returns a single
deduplicated, sorted list of entities ready for the redactors.
"""

from app.detectors.presidio_detector import detect_with_presidio
from app.detectors.local_context_detector import detect_local_context
from app.detectors.masakhaner_detector import detect_with_masakhaner


# ---------------------------------------------------------------------------
# Priority rules
# ---------------------------------------------------------------------------
# When two (or more) detectors flag overlapping spans, this decides which
# one wins. Checked in this order in _pick_winner:
#
#   1. MASAKHANER_PREFERRED_TYPES - MasakhaNER wins for PERSON_NAME over
#      BOTH local_context and Presidio. It's a model trained specifically
#      on Hausa/Igbo/Yoruba-annotated text, so it generalizes to Nigerian
#      names local_context_detector's curated fragment list doesn't cover
#      (e.g. "Oladele" was missing before that fix), and outperforms
#      Presidio/spaCy's English-tuned PERSON recognizer on the same names.
#
#   2. CONTEXT_PREFERRED_TYPES - local_context_detector wins for the
#      remaining free-text entities (LGAs, states of origin, ethnic
#      groups, addresses) where Presidio/spaCy's default English NER
#      isn't tuned for Nigerian-specific terms. Also acts as the PERSON_NAME
#      fallback whenever MasakhaNER doesn't fire on a given span at all
#      (no overlap = no conflict = no priority check needed - it just
#      passes through untouched).
#
#   3. PATTERN_PREFERRED_TYPES - Presidio/regex wins for fixed-format
#      entities (IDs, numbers) since it's deterministic and pattern-based.
#
#   4. Fallback: higher confidence score, then longer span.
#
# SPAN WIDENING ON SAME-TYPE OVERLAP
# ------------------------------------
# CONFIRMED BUG (found via app/debug/pipeline_dump.py against the typed-
# scan test document, after the PSM 6 OCR fix): MasakhaNER's wordpiece
# aggregation can clip a span by a character or two at a boundary - e.g.
# it returned "Chiamaka Ngozi Ez" (missing the final "e") while
# local_context_detector, overlapping the same name, correctly returned
# the full "Chiamaka Ngozi Eze". Under pure source-priority, MasakhaNER
# won unconditionally (per MASAKHANER_PREFERRED_TYPES) and the merged
# result kept the truncated span - meaning the final "e" would have been
# left un-redacted in the output PDF.
#
# This is exactly the partial-overlap case flagged as a known risk below
# (previously: "this returns ONE of the two spans as-is; it does not
# widen a span to cover a partial overlap... worth revisiting if
# partial-overlap cases show up in testing"). One now has, so it's
# implemented: whenever two overlapping detections share the same
# entity_type, the winner's span is widened to the UNION of both spans'
# boundaries before being returned. Source-priority still decides whose
# "source"/score attribution survives; only the character span is
# widened, and only ever wider, never narrower. Overlaps between
# DIFFERENT entity_types are unaffected - widening across type boundaries
# isn't the fix for what was actually seen here and risks conflating two
# genuinely different entities.

MASAKHANER_PREFERRED_TYPES = {
    "PERSON_NAME",
}

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


# Maps MasakhaNER's raw label(s) onto this pipeline's canonical entity
# types. MasakhaNER emits "PERSON" (a Hugging Face NER label), not this
# app's "PERSON_NAME" - without this mapping, _pick_winner()'s priority
# checks against MASAKHANER_PREFERRED_TYPES / CONTEXT_PREFERRED_TYPES
# never match, since they compare entity_type strings directly. That
# silently breaks priority resolution AND is the root cause of duplicate
# PERSON detections surviving into the merged output: two spans that
# really are the same entity (one from MasakhaNER, one from
# local_context) go through _spans_overlap() fine (overlap is purely
# start/end based), but a would-be winner check keyed on entity_type
# never recognizes them as competing for the same slot in the first
# place if downstream code branches on type before calling _pick_winner -
# and more importantly, once normalized, genuinely overlapping spans of
# the same canonical type are far more likely to be caught by
# _spans_overlap() and correctly deduplicated to one winner instead of
# both surviving as separate "different-type" entities.
#
# "PER" is included defensively in case the underlying Hugging Face
# pipeline's aggregation strategy is ever changed to return the shorter
# CoNLL-style tag instead of the expanded "PERSON" label.
MASAKHANER_LABEL_MAP = {
    "PERSON": "PERSON_NAME",
    "PER": "PERSON_NAME",
}


def _normalize_masakhaner_results(masakhaner_results: list[dict]) -> list[dict]:
    """
    Applies MASAKHANER_LABEL_MAP to each result's entity_type. Any label
    not in the map passes through unchanged (defensive default, not a
    silent drop) - so if masakhaner_detector.py is ever updated to emit
    the correct canonical label directly, this becomes a no-op rather
    than something that needs to be un-done.
    """
    normalized = []
    for r in masakhaner_results:
        r = dict(r)  # don't mutate the detector's own return value
        r["entity_type"] = MASAKHANER_LABEL_MAP.get(r["entity_type"], r["entity_type"])
        normalized.append(r)
    return normalized


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------

def _spans_overlap(a: dict, b: dict) -> bool:
    """True if two entity spans share any character position."""
    if a["start"] is None or b["start"] is None:
        return False
    return a["start"] < b["end"] and b["start"] < a["end"]


def _pick_winner(a: dict, b: dict, text: str) -> dict:
    """
    Given two overlapping detections, decide which one to keep.

    Priority order:
    1. MasakhaNER wins for PERSON_NAME (over both local_context and Presidio)
    2. local_context wins for its preferred free-text types (names/places,
       when MasakhaNER isn't the source on either side of the overlap)
    3. Presidio wins for its preferred fixed-format ID types
    4. Higher confidence score
    5. Longer span (more specific match)

    SPAN WIDENING: if both detections share the same entity_type, the
    winner's span is widened to the union of both spans' boundaries
    before being returned (start=min, end=max), and "text" is re-sliced
    from `text` at that widened range. This fixes the confirmed
    truncated-span bug documented above (MasakhaNER clipping the final
    character of a name that local_context correctly found in full) -
    the higher-priority source still wins for "source"/score
    attribution, but never at the cost of narrowing below what a
    lower-priority detector correctly matched. Overlaps between
    different entity_types are returned as-is, unwidened - see the note
    above the priority-rule constants for why.
    """
    a_type, b_type = a["entity_type"], b["entity_type"]

    if a["source"] == "masakhaner" and a_type in MASAKHANER_PREFERRED_TYPES:
        winner = a
    elif b["source"] == "masakhaner" and b_type in MASAKHANER_PREFERRED_TYPES:
        winner = b
    elif a["source"] == "local_context" and a_type in CONTEXT_PREFERRED_TYPES:
        winner = a
    elif b["source"] == "local_context" and b_type in CONTEXT_PREFERRED_TYPES:
        winner = b
    elif a["source"] == "presidio" and a_type in PATTERN_PREFERRED_TYPES:
        winner = a
    elif b["source"] == "presidio" and b_type in PATTERN_PREFERRED_TYPES:
        winner = b
    elif a["score"] != b["score"]:
        winner = a if a["score"] > b["score"] else b
    else:
        a_len = a["end"] - a["start"]
        b_len = b["end"] - b["start"]
        winner = a if a_len >= b_len else b

    if a_type == b_type:
        widened_start = min(a["start"], b["start"])
        widened_end = max(a["end"], b["end"])
        if widened_start != winner["start"] or widened_end != winner["end"]:
            winner = dict(winner)
            winner["start"] = widened_start
            winner["end"] = widened_end
            winner["text"] = text[widened_start:widened_end]

    return winner


def _merge_overlaps(detections: list[dict], text: str) -> list[dict]:
    """
    Sorts detections by start position, then walks through resolving
    any overlapping spans down to a single winner each. `text` is the
    original text detections were found in - needed by _pick_winner()
    to re-slice a widened span's text.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: (d["start"], d["end"]))

    merged = [detections[0]]

    for current in detections[1:]:
        last = merged[-1]

        if _spans_overlap(last, current):
            winner = _pick_winner(last, current, text)
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
    local gazetteer/heuristic context detection + MasakhaNER PERSON
    detection) and returns a single merged, deduplicated list of
    entities sorted by position in the text.

    No external API calls anywhere in this pipeline (MasakhaNER's model
    weights are downloaded once from Hugging Face on first run and
    cached locally after that - no per-request network call).

    Output shape (ready for redactors):
        {
            "text": "...",
            "entity_type": "...",
            "start": N,
            "end": N,
            "score": 0.0-1.0,
            "source": "presidio" | "local_context" | "masakhaner",
        }
    """
    if not text or not text.strip():
        return []

    presidio_results = detect_with_presidio(text, language=language)
    context_raw = detect_local_context(text)
    context_results = _normalize_local_context_results(context_raw)
    masakhaner_raw = detect_with_masakhaner(text)
    masakhaner_results = _normalize_masakhaner_results(masakhaner_raw)

    all_detections = presidio_results + context_results + masakhaner_results
    merged = _merge_overlaps(all_detections, text)

    return merged


# ---------------------------------------------------------------------------
# Inline test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_text = (
        "Applicant: Chukwuemeka Okonkwo, NIN: 12345678901, "
        "BVN: 22334455667, Phone: 08012345678, "
        "Passport: A01234567, Plate: LAG 123AB. "
        "He resides in Nnewi North, Anambra State, and is of Igbo ethnicity. "
        "Employee record on file for Oladele Peter, submitted last month."
    )

    print(f"Testing merger.py against sample text:\n{test_text}\n")

    results = detect_all(test_text)

    if not results:
        print("No entities detected — check that all three detectors are working.")
    else:
        for r in results:
            print(
                f"[{r['source']}] {r['entity_type']}: '{r['text']}' "
                f"(pos {r['start']}-{r['end']}, score {r['score']})"
            )

    # Regression test for the confirmed span-truncation bug: simulates
    # MasakhaNER returning a name span that's one character short of
    # local_context's correct, complete span (as actually observed
    # against the typed-scan test document - MasakhaNER returned
    # "Chiamaka Ngozi Ez" while local_context returned the correct
    # "Chiamaka Ngozi Eze"). Expect the merged result to keep the FULL,
    # untruncated span even though MasakhaNER wins on source priority.
    print("\n--- Span-widening regression check (expect full untruncated name) ---")
    truncation_text = "Full Name: Chiamaka Ngozi Eze"
    truncated_masakhaner = [{
        "text": "Chiamaka Ngozi Ez", "entity_type": "PERSON_NAME",
        "start": 11, "end": 28, "score": 0.97, "source": "masakhaner",
    }]
    correct_local_context = [{
        "text": "Chiamaka Ngozi Eze", "entity_type": "PERSON_NAME",
        "start": 11, "end": 29, "score": 0.85, "source": "local_context",
    }]
    widened = _merge_overlaps(truncated_masakhaner + correct_local_context, truncation_text)
    if len(widened) == 1 and widened[0]["text"] == "Chiamaka Ngozi Eze" and widened[0]["end"] == 29:
        print(f"OK: merged span is full and untruncated: {widened[0]}")
    else:
        print(f"FAIL: expected full untruncated span, got: {widened}")