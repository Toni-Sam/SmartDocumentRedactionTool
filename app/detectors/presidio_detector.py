"""
presidio_detector.py

Base PII detection layer combining:
- Presidio's default recognizers (emails, dates, generic PERSON/ORG via spaCy NER)
- Custom Nigerian pattern recognizers (NIN, BVN, phone, passport, plate)
- Yoruba name gazetteer recognizer (deny-list match against names mined
  from MasakhaNER's Yoruba CoNLL data - catches isolated names with no
  surrounding sentence context, which masakhaner_detector.py's model can
  miss and local_context_detector.py's label/title anchors don't fire on)

This is the first pass of the pipeline. Output feeds into merger.py
alongside local_context_detector.py results (gazetteer + heuristic
name/LGA/state detection - no external API calls).

No functional changes were needed in this file for the Claude API
removal; it never depended on Claude. Docstring updated only to
reflect the new detector name downstream.
"""

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

from app.detectors.nigerian_patterns import (
    nin_recognizer,
    bvn_recognizer,
    passport_recognizer,
    phone_recognizer,
    plate_recognizer,
    vin_recognizer,
    drivers_license_recognizer,
    tin_recognizer,
    nuban_recognizer,
    rsa_pin_recognizer,
    nysc_recognizer,
    cac_rc_recognizer,
    nhis_recognizer,
    yoruba_name_recognizer,
)


# ---------------------------------------------------------------------------
# NLP engine configuration (spaCy)
# ---------------------------------------------------------------------------

NLP_CONFIGURATION = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
}


def _build_analyzer() -> AnalyzerEngine:
    """
    Builds a Presidio AnalyzerEngine with:
    - Default recognizers (EMAIL_ADDRESS, DATE_TIME, PERSON, ORG, LOCATION, etc.)
    - Nigerian custom recognizers (NG_NIN, NG_BVN, NG_PASSPORT, NG_PHONE, NG_PLATE)
    - Yoruba name gazetteer recognizer (PERSON_NAME, deny-list based)
    """
    provider = NlpEngineProvider(nlp_configuration=NLP_CONFIGURATION)
    nlp_engine = provider.create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)

    # Add Nigerian-specific recognizers on top of the defaults
    registry.add_recognizer(nin_recognizer)
    registry.add_recognizer(bvn_recognizer)
    registry.add_recognizer(passport_recognizer)
    registry.add_recognizer(phone_recognizer)
    registry.add_recognizer(plate_recognizer)
    registry.add_recognizer(vin_recognizer)
    registry.add_recognizer(drivers_license_recognizer)
    registry.add_recognizer(tin_recognizer)
    registry.add_recognizer(nuban_recognizer)
    registry.add_recognizer(rsa_pin_recognizer)
    registry.add_recognizer(nysc_recognizer)
    registry.add_recognizer(cac_rc_recognizer)
    registry.add_recognizer(nhis_recognizer)
    registry.add_recognizer(yoruba_name_recognizer)

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
    return analyzer


# Build once at import time — analyzer initialization is expensive (loads spaCy model)
_analyzer = _build_analyzer()


# ---------------------------------------------------------------------------
# Public detection function
# ---------------------------------------------------------------------------

# Entities we actually want back. Presidio's defaults include noisy types
# (URL, IP_ADDRESS, CRYPTO, etc.) that aren't relevant to this project.
#
# PERSON_NAME added here for the new yoruba_name_recognizer - without it,
# TARGET_ENTITIES would silently filter its output out of analyze()'s
# results entirely (Presidio only returns types explicitly requested when
# the `entities` argument is passed). This is separate from "PERSON"
# below, which is spaCy's own default NER label and comes from a
# different recognizer entirely - the two don't collide in merger.py,
# see nigerian_patterns.py's comment on the gazetteer recognizer for why.
TARGET_ENTITIES = [
    "PERSON",
    "PERSON_NAME",
    "LOCATION",
    "DATE_TIME",
    "EMAIL_ADDRESS",
    "NG_NIN",
    "NG_BVN",
    "NG_PASSPORT",
    "NG_PHONE",
    "NG_PLATE",
    "US_SSN",
    "NG_VIN",
    "NG_DRIVERS_LICENSE",
    "NG_TIN",
    "NG_NUBAN",
    "NG_RSA_PIN",
    "NG_NYSC",
    "NG_NYSC_STATE_CODE",
    "NG_CAC_RC",
    "NG_NHIS",
]


def detect_with_presidio(text: str, language: str = "en") -> list[dict]:
    """
    Runs Presidio analysis on the given text and returns normalized results.

    Returns a list of dicts shaped like:
        {
            "text": "12345678901",
            "entity_type": "NG_NIN",
            "start": 14,
            "end": 25,
            "score": 0.6,
            "source": "presidio",
        }

    This shape matches what merger.py expects from every detector, so
    Presidio and local_context_detector results can be merged without
    extra conversion.
    """
    if not text or not text.strip():
        return []

    results = _analyzer.analyze(
        text=text,
        language=language,
        entities=TARGET_ENTITIES,
    )

    normalized = [
        {
            "text": text[r.start:r.end],
            "entity_type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": round(r.score, 2),
            "source": "presidio",
        }
        for r in results
    ]

    return normalized


# ---------------------------------------------------------------------------
# Inline test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_text = (
        "Applicant: Chukwuemeka Okonkwo, NIN: 12345678901, "
        "BVN: 22334455667, Phone: 08012345678, "
        "Passport: A01234567, Plate: LAG 123AB. "
        "Email: chukwuemeka@example.com. Date of birth: 14 March 1990. "
        "Resides in Nnewi, Anambra State. "
        "US SSN (for cross-border applicant): 123-45-6789. "
        "Voter's Identification Number (VIN): 804R DFT9 2617 3628 1038."
    )

    print(f"Testing presidio_detector.py against sample text:\n{test_text}\n")

    detections = detect_with_presidio(test_text)

    if not detections:
        print("No entities detected — check recognizer registration.")
    else:
        for d in detections:
            print(
                f"[{d['source']}] {d['entity_type']}: '{d['text']}' "
                f"(pos {d['start']}-{d['end']}, score {d['score']})"
            )

    # New: isolated-name check with no title, no label, no sentence
    # context at all - the exact case that started this whole recognizer.
    # Uses a Yoruba name from the gazetteer that is deliberately NOT in
    # local_context_detector.py's NIGERIAN_NAME_FRAGMENTS list, so this
    # test actually isolates the new recognizer's contribution rather
    # than passing coincidentally via the other layer.
    print("\n--- Isolated-name check (no context, name not in local_context's fragment list) ---")
    isolated_text = "Adejumobi"
    isolated_detections = [
        d for d in detect_with_presidio(isolated_text) if d["entity_type"] == "PERSON_NAME"
    ]
    if isolated_detections:
        for d in isolated_detections:
            print(f"[{d['source']}] {d['entity_type']}: '{d['text']}' (score {d['score']})")
    else:
        print("FAIL: gazetteer recognizer did not fire — check TARGET_ENTITIES/registration.")