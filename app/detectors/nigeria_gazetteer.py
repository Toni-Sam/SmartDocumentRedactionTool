"""
nigeria_gazetteer.py

Local, offline replacement for the part of claude_detector.py that
identified LGAs, states of origin, and ethnic groups from context.

Instead of asking an LLM to infer these, this module does deterministic
substring matching against known, finite lists:
- All 774 Nigerian LGAs (verified against the official per-state counts:
  774 total, e.g. Kano=44, Katsina=34, Bayelsa=8 - see nigeria_gazetteer.json)
- All 36 states + the Federal Capital Territory
- A curated (non-exhaustive) list of major Nigerian ethnic groups

This trades some recall for zero API cost and 100% determinism: it will
never hallucinate an LGA that doesn't exist, but it also will not catch
a misspelled LGA name. If you need typo-tolerance later, rapidfuzz can
be layered on top of ALL_LGAS / ALL_STATES without changing the public
functions below.

Data source: app/data/nigeria_gazetteer.json, built from a public,
verified Nigerian LGA dataset (774 LGAs / 37 states incl. FCT).
"""

import json
import re
from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "nigeria_gazetteer.json"

with open(_DATA_PATH, "r", encoding="utf-8") as f:
    _RAW = json.load(f)

STATES: list[str] = _RAW["states"]  # 36 states + Federal Capital Territory
LGAS_BY_STATE: dict[str, list[str]] = _RAW["lgas_by_state"]

ALL_LGAS: list[str] = sorted(
    {lga for lgas in LGAS_BY_STATE.values() for lga in lgas}
)

LGA_TO_STATE: dict[str, str] = {
    lga: state for state, lgas in LGAS_BY_STATE.items() for lga in lgas
}

# Common informal names for the FCT that won't appear verbatim in the
# official state list but show up constantly in real documents.
STATE_ALIASES = {
    "Abuja": "Federal Capital Territory",
    "FCT": "Federal Capital Territory",
    "FCT Abuja": "Federal Capital Territory",
}
ALL_STATE_TERMS: list[str] = sorted(set(STATES) | set(STATE_ALIASES.keys()))

# Not exhaustive (Nigeria has 250+ recognized ethnic groups) - this
# covers the major/most commonly self-declared ones on official forms.
# Extend this list as needed; it's just a flat list, no code changes
# required elsewhere.
ETHNIC_GROUPS: list[str] = [
    "Yoruba", "Igbo", "Hausa", "Fulani", "Ijaw", "Kanuri", "Ibibio",
    "Tiv", "Edo", "Bini", "Nupe", "Igala", "Idoma", "Urhobo", "Itsekiri",
    "Efik", "Annang", "Ekoi", "Gwari", "Gbagyi", "Jukun", "Ogoni",
    "Isoko", "Ebira", "Esan", "Ijebu", "Egba", "Kalabari", "Ikwerre",
    "Ogba", "Etsako", "Chamba", "Bachama", "Angas", "Berom", "Waja",
    "Shuwa Arab", "Yala", "Ikwo", "Ishan",
]


def _compile_boundary_pattern(terms: list[str]) -> re.Pattern:
    """
    Builds one compiled regex from a list of terms, longest first (so
    multi-word LGAs like "Isiala Ngwa North" match before the shorter
    "Isiala Ngwa" would, and no such prefix collision goes undetected).
    Word-boundaried and case-insensitive.
    """
    escaped = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


_LGA_PATTERN = _compile_boundary_pattern(ALL_LGAS)
_STATE_PATTERN = _compile_boundary_pattern(ALL_STATE_TERMS)
_ETHNIC_PATTERN = _compile_boundary_pattern(ETHNIC_GROUPS)


def find_lgas(text: str) -> list[dict]:
    """Returns [{"text", "start", "end", "state"}] for every LGA match."""
    matches = []
    for m in _LGA_PATTERN.finditer(text):
        matched = m.group(0)
        # Recover canonical casing/state via case-insensitive lookup.
        canonical = next(
            (lga for lga in ALL_LGAS if lga.lower() == matched.lower()), matched
        )
        matches.append({
            "text": matched,
            "start": m.start(),
            "end": m.end(),
            "state": LGA_TO_STATE.get(canonical, "UNKNOWN"),
        })
    return matches


def find_states(text: str) -> list[dict]:
    """Returns [{"text", "start", "end"}] for every state/FCT-alias match."""
    return [
        {"text": m.group(0), "start": m.start(), "end": m.end()}
        for m in _STATE_PATTERN.finditer(text)
    ]


def find_ethnic_groups(text: str) -> list[dict]:
    """Returns [{"text", "start", "end"}] for every ethnic group match."""
    return [
        {"text": m.group(0), "start": m.start(), "end": m.end()}
        for m in _ETHNIC_PATTERN.finditer(text)
    ]


# ---------------------------------------------------------------------------
# Inline test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loaded {len(ALL_LGAS)} LGAs across {len(STATES)} states/FCT.\n")

    test_text = (
        "Applicant resides in Nnewi North, Anambra State, and is of Igbo "
        "ethnicity. Alternative contact lives in Isiala Ngwa North, Abia "
        "State. Office located in Abuja, FCT. Igala and Idoma are also mentioned."
    )

    print(f"Testing nigeria_gazetteer.py against sample text:\n{test_text}\n")

    for label, results in [
        ("LGA", find_lgas(test_text)),
        ("STATE", find_states(test_text)),
        ("ETHNIC_GROUP", find_ethnic_groups(test_text)),
    ]:
        if not results:
            print(f"{label}: no matches")
            continue
        for r in results:
            extra = f" (state: {r['state']})" if "state" in r else ""
            recovered = test_text[r["start"]:r["end"]]
            print(f"{label}: '{r['text']}' (pos {r['start']}-{r['end']}){extra} "
                  f"{'OK' if recovered == r['text'] else 'MISMATCH'}")