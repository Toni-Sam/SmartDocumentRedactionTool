"""
masakhaner_detector.py
-----------------------
Detects PERSON entities using a pretrained multilingual NER model
fine-tuned on MasakhaNER (Adelani et al., 2021) - a manually annotated
Named Entity Recognition corpus covering ten African languages,
including Hausa, Igbo, and Yorùbá, three of Nigeria's largest.

WHY THIS EXISTS
----------------
local_context_detector.py's find_person_names() depends on a curated
list of Nigerian name fragments (NIGERIAN_NAME_FRAGMENTS). That approach
can never keep up with the actual space of Nigerian given/family names -
the vocabulary spans thousands of roots across dozens of ethnic groups,
and any hand-curated list will always miss common, legitimate names
(as "oladele" being absent demonstrated).

This module takes a different approach entirely: instead of matching
against a known list, it uses a model that has *learned the patterns*
of Hausa/Igbo/Yorùbá names from real annotated text, the same way
spaCy's English model recognizes "Sarah Thompson" without "Sarah" or
"Thompson" being hardcoded anywhere. That generalizes to names never
seen before, which a list fundamentally cannot do.

MODEL
-----
Davlan/bert-base-multilingual-cased-masakhaner - an mBERT model
fine-tuned on the MasakhaNER dataset (github.com/masakhane-io/masakhane-ner).
It recognizes PER, ORG, LOC, and DATE entities across ten African
languages. Only PER is used here; ORG/LOC/DATE are intentionally
discarded since presidio_detector.py and local_context_detector.py
already cover those categories through other means (gazetteer, Presidio
defaults).

Downloaded once from Hugging Face on first run (several hundred MB) and
cached locally afterward (~/.cache/huggingface by default) - no API key,
no per-request cost, no network call needed after the first download.
Consistent with this project's "no external API calls" constraint (the
same reason the Claude API layer was removed).

A larger, generally more accurate alternative exists at
"Davlan/xlm-roberta-large-masakhaner" if this model's recall turns out
too low in testing - same interface, just swap MODEL_NAME.

CAVEAT - UNVALIDATED ON ENGLISH-LANGUAGE DOCUMENTS
-----------------------------------------------------
MasakhaNER's training text is Hausa/Igbo/Yorùbá news text, not English.
Your actual documents are primarily English prose with Nigerian names
embedded in them (e.g. "Applicant: Oladele Peter"). mBERT's multilingual
pretraining gives it cross-lingual representations, so it *may* transfer
reasonably to recognizing Nigerian names inside English sentences, but
that transfer is not guaranteed and hasn't been measured against your
actual sample documents yet. Treat this module as a candidate additional
signal for merger.py, not a proven replacement for the existing name
heuristics - validate recall and false-positive rate against your real
test documents (run the __main__ block below, then try it against
tests/NigerianSamples/ content) before wiring it into merger.py with a
priority that overrides the other detectors.

DEPENDENCIES
------------
Not yet part of this project's requirements. Install with:
    pip install transformers torch --break-system-packages

First run will download the model weights - requires internet access at
least once. After that, it runs fully offline from the local cache.

USAGE
-----
    from app.detectors.masakhaner_detector import detect_with_masakhaner

    entities = detect_with_masakhaner("Applicant: Oladele Peter, BVN: ...")
    # -> [{"text": "Oladele Peter", "entity_type": "PERSON_NAME",
    #      "start": 11, "end": 24, "score": 0.97, "source": "masakhaner"}, ...]

Output shape matches every other detector in this pipeline
(presidio_detector.py, local_context_detector.py), so it can be added
to merger.py's detect_all() later by adding one more call + one more
normalization step, the same pattern local_context_detector already
follows. Not wired in yet - this module is standalone until its
accuracy has been validated on your documents.
"""

from transformers import pipeline

MODEL_NAME = "Davlan/bert-base-multilingual-cased-masakhaner"

# Built once at import time - loading the model is expensive (downloads
# on first run, then loads weights into memory on every subsequent run),
# same reasoning as presidio_detector.py's module-level _analyzer.
_ner_pipeline = pipeline(
    "ner",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    aggregation_strategy="simple",  # merges sub-word tokens into whole-entity
                                     # spans and returns real character offsets
)


def detect_with_masakhaner(text: str) -> list[dict]:
    """
    Runs the MasakhaNER-based model against `text` and returns only PER
    (person) entities, normalized to this pipeline's common shape.

    Returns:
        [{"text": ..., "entity_type": "PERSON_NAME", "start": ..., "end": ...,
          "score": ..., "source": "masakhaner"}, ...]
    """
    if not text or not text.strip():
        return []

    raw_results = _ner_pipeline(text)

    entities = []
    for r in raw_results:
        if r.get("entity_group") != "PER":
            continue
        start, end = r["start"], r["end"]
        entities.append({
            # Re-sliced from the source text rather than trusting r["word"],
            # which can carry tokenizer artifacts (e.g. leftover "##" pieces
            # or normalized casing) instead of the original substring.
            "text": text[start:end],
            "entity_type": "PERSON_NAME",
            "start": start,
            "end": end,
            "score": round(float(r["score"]), 2),
            "source": "masakhaner",
        })

    return entities


if __name__ == "__main__":
    test_texts = [
        # English prose with an embedded Nigerian name and no title/label
        # anchor - the exact case local_context_detector.py was missing
        # before the "oladele" fragment-list fix.
        "Employee record on file for Oladele Peter, submitted last month.",

        # Names with no overlap with local_context_detector's curated
        # fragment list at all, to test genuine generalization rather
        # than memorized vocabulary.
        "Applicant: Ngozi Eberechi Danjuma, next of kin: Yemi Osinachi.",

        # Mixed indigenous + English names in a form-like structure.
        "Full Name: Babatunde Chukwuemeka. Guarantor: Halima Abdulrahman.",
    ]

    print(f"Testing masakhaner_detector.py (model: {MODEL_NAME})\n")

    for text in test_texts:
        print(f"Text: {text}")
        results = detect_with_masakhaner(text)
        if not results:
            print("  No PERSON entities detected.")
        else:
            for e in results:
                print(f"  [{e['entity_type']}] '{e['text']}' "
                      f"(pos {e['start']}-{e['end']}, score {e['score']}, source={e['source']})")
        print()