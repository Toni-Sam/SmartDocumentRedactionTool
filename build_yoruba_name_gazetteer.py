"""
build_yoruba_name_gazetteer.py

One-off build script: mines PER-tagged spans from MasakhaNER's Yoruba
CoNLL splits (1.0 + 2.0) and writes a cleaned, deduplicated name
gazetteer to app/data/yoruba_names_gazetteer.json.

Source: MasakhaNER (Adelani et al., 2021 - TACL; Adelani et al., 2022 -
EMNLP). Dataset license: CC-BY-NC-4.0 (non-commercial use only).
Repo: https://github.com/masakhane-io/masakhane-ner

NOTE ON SCOPE: these are PER entities as they occur in Yoruba-language
news text, not a curated list of ethnically-Yoruba names. Coverage
includes non-Yoruba public figures who appear in Yoruba-language news
(e.g. "Buhari"). For this project's purpose - catching Nigerian names
the other detectors miss - that's treated as acceptable extra recall,
not noise to filter out.

Run as a module from the project root:
    python -m scripts.build_yoruba_name_gazetteer
"""

import json
import re
import unicodedata
from pathlib import Path

# Paths to the raw MasakhaNER CoNLL files, once downloaded locally.
# Adjust MASAKHANER_ROOT if you clone the repo elsewhere.
MASAKHANER_ROOT = Path("masakhane-ner")

SOURCE_FILES = [
    MASAKHANER_ROOT / "data" / "yor" / "train.txt",           # MasakhaNER 1.0
    MASAKHANER_ROOT / "data" / "yor" / "dev.txt",
    MASAKHANER_ROOT / "data" / "yor" / "test.txt",
    MASAKHANER_ROOT / "MasakhaNER2.0" / "data" / "yor" / "train.txt",  # MasakhaNER 2.0
    MASAKHANER_ROOT / "MasakhaNER2.0" / "data" / "yor" / "dev.txt",
    MASAKHANER_ROOT / "MasakhaNER2.0" / "data" / "yor" / "test.txt",
]

OUTPUT_PATH = Path("app") / "data" / "yoruba_names_gazetteer.json"

# Minimum length (after cleaning) for a single-token entry to be kept.
# Raised from an initial 2 to 4 after testing surfaced real false
# positives: at MIN_LENGTH=2, common short words like "The", "Mr", "Sam",
# "May", "Van", "Son" ended up in the gazetteer as single-token PER
# artifacts and matched constantly in ordinary prose (deny_list matching
# is case-insensitive - "the" matched ANY occurrence of "the"). Raising
# to 4 removes the great majority of that noise. Only applied to
# single-token entries; multi-word full-name spans keep no minimum,
# since a 2-word capitalized phrase is far less likely to coincide with
# ordinary prose by chance.
MIN_LENGTH = 4

# Deliberately NOT a general English dictionary filter - tried that
# during testing and it excluded genuinely correct Nigerian given names
# (James, David, Grace, Comfort, Success, etc.) that local_context_detector.py's
# own NIGERIAN_NAME_FRAGMENTS list explicitly accepts as valid names
# despite being ordinary English words too. That's an accepted,
# pre-existing tradeoff in this project, not something to fix here.
#
# This stoplist instead targets two narrower, higher-confidence
# categories found by inspecting the actual extracted output:
#   1. Grammatical function words (articles/prepositions/conjunctions)
#      that are never names under any circumstance.
#   2. Specific entries confirmed to be NER mistags in the source corpus
#      - place names (Ohio, Georgia, Montana, Israel, Chile, Guinea) and
#      role/description nouns (Advocate, Commentator, Language, Court)
#      that got PER-tagged incorrectly, not names of any kind.
# This is a reduction of the worst false-positive risk, not exhaustive
# precision - some residual risk remains for real names that are also
# ordinary words (e.g. "Grace", "Hope"), same as it already does
# elsewhere in this pipeline. Revisit based on real-document testing.
COMMON_WORD_STOPLIST = {
    # Function words
    "the", "and", "for", "are", "was", "were", "with", "from", "this",
    "that", "have", "has", "had", "not", "but", "you", "your", "our",
    "his", "her", "its", "who", "what", "when", "where", "why", "how",
    # Confirmed NER mistags: places
    "ohio", "georgia", "montana", "israel", "chile", "guinea", "caribbean",
    # Confirmed NER mistags: roles/descriptions/generic nouns
    "advocate", "commentator", "language", "court", "grange",
    "possibility", "rainbow", "macaroni", "fabulous", "witty", "skinny",
    "slows", "sale", "fall", "west", "east", "north", "south", "bush",
    "gold", "long", "blue", "easy", "iron", "lady", "lone", "black",
    "white",
}


def extract_per_spans(path: Path) -> set[str]:
    """
    Walks one CoNLL-format file and reconstructs each contiguous
    B-PER (I-PER)* run into a single name span.
    """
    spans: set[str] = set()
    current: list[str] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) != 2:
                continue
            token, tag = parts

            if tag == "B-PER":
                if current:
                    spans.add(" ".join(current))
                current = [token]
            elif tag == "I-PER" and current:
                current.append(token)
            else:
                if current:
                    spans.add(" ".join(current))
                current = []

    if current:
        spans.add(" ".join(current))

    return spans


def strip_diacritics(span: str) -> str:
    """
    Strips Yoruba tonal/underdot diacritics (e.g. "Adéjùmọ̀bí" -> "Adejumobi"),
    normalizing to the plain-ASCII spelling real documents actually use.

    WHY THIS MATTERS: MasakhaNER's source text preserves full Yoruba
    orthography, but this project's target documents (English-language
    forms/prose) are typed on standard keyboards - diacritics essentially
    never appear. Without this step, ~40% of the raw extracted gazetteer
    (every name that happened to include a tone mark) would never match
    a real document, since deny_list matching is exact-string, not
    diacritic-insensitive. Confirmed empirically during testing: "Adejumobi"
    (as it would actually appear typed) had zero matches against the raw
    extraction, because only the diacritic form "ADÉJÙMỌ̀BÍ" was present.
    """
    decomposed = unicodedata.normalize("NFD", span)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def clean_span(span: str, min_length: int) -> str | None:
    """
    Cleans a raw extracted span:
    - collapses " - " (from hyphenated names split across tokens by the
      CoNLL tokenizer) back into "-" with no surrounding spaces
    - strips leading/trailing punctuation
    - rejects spans that are empty, too short, pure punctuation, or hit
      the common-word stoplist after cleaning

    min_length is passed in rather than hardcoded, since it differs
    between full multi-word spans (no minimum - see MIN_LENGTH comment
    above) and individual single tokens (MIN_LENGTH applies).
    """
    cleaned = re.sub(r"\s*-\s*", "-", span)
    cleaned = cleaned.strip(" .,;:'\"()-")

    if len(cleaned) < min_length:
        return None
    if not any(c.isalpha() for c in cleaned):
        return None
    if cleaned.lower() in COMMON_WORD_STOPLIST:
        return None

    return cleaned


def main() -> None:
    all_raw_spans: set[str] = set()

    for path in SOURCE_FILES:
        if not path.exists():
            print(f"WARNING: {path} not found, skipping.")
            continue
        found = extract_per_spans(path)
        print(f"{path}: {len(found)} unique raw PER spans")
        all_raw_spans |= found

    cleaned_full_spans: set[str] = set()
    cleaned_tokens: set[str] = set()

    for span in all_raw_spans:
        # Basic punctuation/hyphen cleanup first, no length filtering yet -
        # needed to know whether this span is single-word or multi-word
        # BEFORE deciding which minimum length rule applies to it.
        pre_cleaned = re.sub(r"\s*-\s*", "-", span).strip(" .,;:'\"()-")
        if not pre_cleaned or not any(c.isalpha() for c in pre_cleaned):
            continue

        is_multi_word = " " in pre_cleaned
        span_min_length = 1 if is_multi_word else MIN_LENGTH

        cleaned = clean_span(pre_cleaned, min_length=span_min_length)
        if cleaned is None:
            continue
        cleaned_full_spans.add(cleaned)

        for token in cleaned.split():
            tok = clean_span(token, min_length=MIN_LENGTH)
            if tok is not None:
                cleaned_tokens.add(tok)

    # Gazetteer includes both full multi-token spans (so "Yẹmí Òṣínbàjò"
    # matches as one hit) and individual name tokens (so "Òṣínbàjò" alone,
    # e.g. in a signature line, still matches even without the first name
    # next to it).
    combined = cleaned_full_spans | cleaned_tokens

    # Add diacritic-stripped forms of every entry (see strip_diacritics()
    # docstring - this is the fix for the confirmed real-document match
    # gap, not a nice-to-have). Originals are kept too: harmless if a
    # document genuinely does use diacritics, and costs nothing extra at
    # match time since this is exact deny_list lookup, not a scan cost
    # per entry.
    stripped = {strip_diacritics(name) for name in combined}
    gazetteer = sorted(combined | stripped)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(gazetteer, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(gazetteer)} entries to {OUTPUT_PATH}")
    print(f"  ({len(cleaned_full_spans)} full spans, {len(cleaned_tokens)} individual tokens, "
          f"before overlap dedup between the two sets)")


if __name__ == "__main__":
    main()