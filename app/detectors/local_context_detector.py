"""
local_context_detector.py

Local, offline replacement for claude_detector.py. Catches the same
categories the Claude layer used to catch - Nigerian names, LGAs,
states of origin, ethnic markers - without any paid API call.

Two techniques, both deterministic (no LLM):

1. Gazetteer matching (nigeria_gazetteer.py) for LGA, STATE_OF_ORIGIN,
   and ETHNIC_GROUP. These are finite, known lists, so exact matching
   is actually MORE reliable here than an LLM guess - it can't
   hallucinate an LGA that doesn't exist.

2. Heuristic name detection for PERSON_NAME. This is the part that
   genuinely loses accuracy without Claude - free-form Nigerian names
   have no finite list. The heuristic below combines:
   - Nigerian title/honorific markers (Mr., Mrs., Chief, Alhaji, Engr., etc.)
   - Common form-field labels ("Name:", "Applicant:", "Next of kin:")
   - A curated list of common Nigerian given/family name fragments,
     used to boost confidence on capitalized word sequences that
     spaCy's default English NER (used elsewhere in the pipeline)
     tends to miss or mis-tag.

   This will have LOWER RECALL than the Claude layer did, especially on
   names with no title/label nearby and no fragment overlap with the
   curated list. See KNOWN_LIMITATIONS.md (or the chat where this was
   generated) for what to do if that recall gap matters for your use case:
   the realistic next step is a small labeled dataset + fine-tuned spaCy
   NER model, not a bigger hardcoded name list.

ADDRESS detection is best-effort only (regex heuristic) and is
intentionally scored low - free-form Nigerian addresses are the
category Claude was doing the most real inference work on, and no
regex substitute is trustworthy enough to score highly here.
"""

import re

from app.detectors.nigeria_gazetteer import find_lgas, find_states, find_ethnic_groups

# ---------------------------------------------------------------------------
# Name detection
# ---------------------------------------------------------------------------

TITLES = [
    "Mr", "Mrs", "Miss", "Ms", "Dr", "Engr", "Barr", "Prof", "Chief",
    "Alhaji", "Alhaja", "Prince", "Princess", "Otunba", "Hon", "Rev",
    "Pastor", "Comrade", "Reverend", 
]
_TITLE_PATTERN = "|".join(re.escape(t) for t in TITLES)

FIELD_LABELS = [
    "Name", "Applicant", "Full Name", "Next of Kin", "Beneficiary",
    "Guarantor", "Employee", "Employer", "Signatory", "Witness",
]
_LABEL_PATTERN = "|".join(re.escape(l) for l in FIELD_LABELS)

# A capitalized word sequence (1-4 words), used as the candidate shape
# for a person name once a title or label context anchors it. The
# between-word separator is deliberately [ \t]+ (same line only), not
# \s+ - a real name doesn't span a line break on a form, but allowing \s+
# here meant a name captured after a label could keep matching straight
# across the newline into the START OF THE NEXT FIELD'S LABEL (e.g.
# "Comfort Adewale\nHome Address" would be captured as one name). Same
# underlying issue as the colon-gap fix above, just on the other side of
# the captured name.
_NAME_SHAPE = r"[A-Z][a-zA-Z'\-]+(?:[ \t]+[A-Z][a-zA-Z'\-]+){0,3}"

# Title/honorific immediately before a name: "Mr. Chukwuemeka Okonkwo"
_TITLE_NAME_PATTERN = re.compile(
    rf"\b(?:{_TITLE_PATTERN})\.?\s+({_NAME_SHAPE})", re.IGNORECASE
)

# Field label followed by a colon then a name: "Applicant: Chukwuemeka Okonkwo"
# The gap between colon and name is deliberately restricted to SAME-LINE
# whitespace ([ \t]*, not \s*) - not newlines. Using \s* here originally
# meant a BLANK field (label with nothing filled in, common on scanned/
# handwritten forms) would let the pattern skip straight across the
# newline and capture the NEXT line's own field label as if it were the
# answer to this one - e.g. "Mother's Name:\nHome Address:" would
# misattribute "Home Address" as a PERSON_NAME. Restricting to same-line
# whitespace means a blank field simply produces no match at all, which
# is the correct behavior - no name was actually written there.
_LABEL_NAME_PATTERN = re.compile(
    rf"\b(?:{_LABEL_PATTERN})[ \t]*:[ \t]*({_NAME_SHAPE})", re.IGNORECASE
)

# Not exhaustive - common Nigerian given/family name fragments used to
# boost confidence on bare capitalized sequences with no title/label
# nearby. Covers frequent Igbo, Yoruba, Hausa/Fulani patterns.
NIGERIAN_NAME_FRAGMENTS = {
    # Igbo
    "chukwu", "chukwuemeka", "chidi", "chidinma", "ngozi", "obiora",
    "uche", "ifeoma", "emeka", "nnamdi", "okonkwo", "eze", "adaeze",
    "chinwe", "ikenna", "obinna", "chioma", "amaka", "nkechi", "kelechi",
    "okoro", "onyekachi", "chiamaka", "okafor", "nwosu", "anyanwu",
    # Yoruba
    "adebayo", "adeyemi", "oluwaseun", "folake", "olamide", "temitope",
    "adewale", "oluwafemi", "bunmi", "yetunde", "babatunde", "ayodele",
    "olusegun", "abiodun", "oyinlola", "adeola", "kehinde", "taiwo",
    "olawale", "adekunle", "ogundipe", "afolabi", "ade", "oladele", "omitogun",
    "oluwatoni", "adebayo", "OMITOGUN"
    # Hausa / Fulani
    "abdullahi", "ibrahim", "musa", "aisha", "fatima", "yusuf", "sani",
    "bello", "aliyu", "hassan", "hussaini", "muhammad", "mohammed",
    "abubakar", "garba", "usman", "suleiman", "zainab", "hadiza",
    "amina", "yakubu",
    # Common Nigerian Christian/English given names - used broadly across
    # ethnic groups, usually paired with an indigenous surname or vice
    # versa ("Oladele Peter", "Emeka Daniel", "Grace Adewale"). Without
    # these, any name where only one word is ethnic-specific and the
    # other is a Western/biblical given name scores 0 on that word and
    # can fall under the 0.5 fragment threshold entirely.
    "peter", "john", "james", "paul", "mary", "grace", "victoria",
    "comfort", "blessing", "joseph", "daniel", "david", "samuel",
    "emmanuel", "elizabeth", "faith", "peace", "precious", "patience",
    "joy", "praise", "goodness", "esther", "ruth", "deborah", "victor",
}


def _fragment_score(candidate: str) -> float:
    """Fraction of words in the candidate that hit the fragment list."""
    words = [w.lower().strip("'-") for w in candidate.split()]
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in NIGERIAN_NAME_FRAGMENTS)
    return hits / len(words)


def _looks_like_stopword_phrase(candidate: str) -> bool:
    """Filters out obvious non-names that match the capitalized shape
    (e.g. sentence-starting words, month names)."""
    STOP = {
        "The", "This", "That", "These", "Those", "January", "February",
        "March", "April", "May", "June", "July", "August", "September",
        "October", "November", "December", "State", "Nigeria", "Federal",
    }
    first_word = candidate.split()[0]
    return first_word in STOP


# Defense-in-depth against form/label text being misclassified as a
# name - e.g. "Home Address", "Local Government", "Public Seal". This
# catches cases beyond the specific blank-field-newline-skip bug fixed
# above (e.g. two field labels sitting adjacent on the same OCR line due
# to layout/reading-order quirks). Deliberately a closed, small
# vocabulary of known form/certificate words, not a general dictionary -
# a real name is extremely unlikely to consist ENTIRELY of words drawn
# from this list.
_ADMIN_VOCABULARY = {
    w.lower() for w in FIELD_LABELS
} | {
    "home", "address", "local", "government", "area", "certificate",
    "origin", "seal", "public", "father", "mother", "state", "date",
    "signature", "hand", "year", "district", "council", "ward",
}


def _is_administrative_phrase(candidate: str) -> bool:
    """True if every word in `candidate` (lowercased, possessive-stripped)
    is drawn from _ADMIN_VOCABULARY - i.e. it looks like form boilerplate,
    not a person's name."""
    words = [w.lower().rstrip("'s").strip("'-") for w in candidate.split()]
    return bool(words) and all(w in _ADMIN_VOCABULARY for w in words)


def find_person_names(text: str) -> list[dict]:
    """
    Returns [{"text", "start", "end", "score"}] for candidate person
    names, using title/label anchoring and Nigerian name-fragment
    scoring. No character-offset issue here (regex gives real offsets
    directly - unlike the old Claude layer, nothing needs to be
    recomputed post-hoc).
    """
    candidates: dict[tuple[int, int], dict] = {}

    for pattern, base_score in (
        (_TITLE_NAME_PATTERN, 0.75),
        (_LABEL_NAME_PATTERN, 0.7),
    ):
        for m in pattern.finditer(text):
            name = m.group(1)
            if _looks_like_stopword_phrase(name) or _is_administrative_phrase(name):
                continue
            start, end = m.start(1), m.end(1)
            score = base_score + 0.15 * _fragment_score(name)
            candidates[(start, end)] = {
                "text": name, "start": start, "end": end,
                "score": min(score, 0.95),
            }

    # Bare capitalized sequences with strong fragment overlap and no
    # title/label nearby - lower confidence, since there's no anchor.
    for m in re.finditer(_NAME_SHAPE, text):
        span = (m.start(), m.end())
        if span in candidates:
            continue
        name = m.group(0)
        if _looks_like_stopword_phrase(name) or _is_administrative_phrase(name) or len(name.split()) < 2:
            continue
        frag_score = _fragment_score(name)
        if frag_score >= 0.5:
            candidates[span] = {
                "text": name, "start": span[0], "end": span[1],
                "score": 0.5 + 0.2 * frag_score,
            }

    return sorted(candidates.values(), key=lambda d: d["start"])


# ---------------------------------------------------------------------------
# Address detection (best-effort, intentionally low confidence)
# ---------------------------------------------------------------------------

_ADDRESS_PATTERN = re.compile(
    r"\b\d{1,4}[A-Za-z]?,?\s+[A-Z][a-zA-Z\s]{2,40}"
    r"(?:Road|Street|Close|Avenue|Crescent|Drive|Lane|Way)\b",
)


def find_addresses(text: str) -> list[dict]:
    return [
        {"text": m.group(0).strip(), "start": m.start(), "end": m.end(), "score": 0.5}
        for m in _ADDRESS_PATTERN.finditer(text)
    ]


# ---------------------------------------------------------------------------
# Public entry point - same output shape as the old claude_detector.py
# so merger.py only needs its import + variable names updated, not its
# core merge logic.
# ---------------------------------------------------------------------------

def detect_local_context(text: str) -> list[dict]:
    """
    Returns a list of entities:
        {"text": "...", "type": "...", "start": N, "end": N, "score": 0.0-1.0}

    Replaces detect_with_claude(). No network call, no API key, no cost.
    """
    if not text or not text.strip():
        return []

    results: list[dict] = []

    for lga in find_lgas(text):
        results.append({
            "text": lga["text"], "type": "LGA",
            "start": lga["start"], "end": lga["end"], "score": 0.85,
        })

    for state in find_states(text):
        results.append({
            "text": state["text"], "type": "STATE_OF_ORIGIN",
            "start": state["start"], "end": state["end"], "score": 0.85,
        })

    for ethnic in find_ethnic_groups(text):
        results.append({
            "text": ethnic["text"], "type": "ETHNIC_GROUP",
            "start": ethnic["start"], "end": ethnic["end"], "score": 0.8,
        })

    for name in find_person_names(text):
        results.append({
            "text": name["text"], "type": "PERSON_NAME",
            "start": name["start"], "end": name["end"], "score": name["score"],
        })

    for addr in find_addresses(text):
        results.append({
            "text": addr["text"], "type": "ADDRESS",
            "start": addr["start"], "end": addr["end"], "score": addr["score"],
        })

    return sorted(results, key=lambda d: d["start"])


# ---------------------------------------------------------------------------
# Inline test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_text = (
        "His firstname and lastname is Chukwuemeka Okonkwo, resides in Nnewi North, Anambra State, "
        "and is of Igbo ethnicity. His brother, Mr. Babatunde Adewale, "
        "lives at 14 Awolowo Road, Ikeja, Lagos."
    )

    print(f"Testing local_context_detector.py against sample text:\n{test_text}\n")

    detections = detect_local_context(test_text)

    if not detections:
        print("No entities detected.")
    else:
        for d in detections:
            recovered = test_text[d["start"]:d["end"]]
            match_ok = recovered == d["text"]
            print(
                f"{d['type']}: '{d['text']}' (pos {d['start']}-{d['end']}, "
                f"score {d['score']:.2f}) "
                f"{'OK' if match_ok else 'MISMATCH: got ' + repr(recovered)}"
            )

    # Regression test: mixed indigenous + Western given-name pair with no
    # title/label anchor nearby, previously missed because neither word
    # individually cleared the 0.5 fragment-score threshold.
    print("\nRegression check: bare mixed-name pair with no title/label anchor")
    mixed_name_text = "Employee record on file for Oladele Peter, submitted last month."
    mixed_detections = [d for d in detect_local_context(mixed_name_text) if d["type"] == "PERSON_NAME"]
    if any(d["text"] == "Oladele Peter" for d in mixed_detections):
        print("OK: 'Oladele Peter' detected.")
    else:
        print(f"STILL MISSING. Detections found: {mixed_detections}")