"""
Structural AI patterns — sentence/paragraph-level tells.

Sources:
- Rosenfeld & Lazebnik (2024): Sentence length uniformity, connector density
- Wikipedia "Signs of AI writing": Rule-of-three, negative parallelism
"""

from __future__ import annotations

import re
import statistics

from core.types import PatternCategory, PatternMatch, Severity


# Discourse connectors that AI overuses
DISCOURSE_CONNECTORS = {
    "moreover", "furthermore", "additionally", "consequently",
    "nevertheless", "nonetheless", "on the other hand", "in addition",
    "as a result", "therefore", "thus", "hence", "meanwhile",
    "subsequently", "alternatively", "conversely", "similarly",
    "likewise", "in contrast", "on the contrary",
}

# "Rule of three" pattern: X, Y, and Z
RULE_OF_THREE_RE = re.compile(
    r'\b(\w+(?:\s+\w+)?),\s+(\w+(?:\s+\w+)?),\s+and\s+(\w+(?:\s+\w+)?)\b',
    re.IGNORECASE,
)

# Negative parallelism: "Not only X but also Y" / "It's not just about X, it's Y"
NEGATIVE_PARALLELISM_RE = re.compile(
    r'\b(not only\b.+?\bbut\s+(?:also\b)?|'
    r"it(?:'s| is) not just about\b.+?,\s*it(?:'s| is)\b)",
    re.IGNORECASE,
)

# Em dash overuse detector
EM_DASH_RE = re.compile(r'[\u2014]|(?:^|\s)--(?:\s|$)')

# Elegant variation: same concept referred to by 3+ different names in close range
# Detect inline-header list: "**Header:** description" pattern
INLINE_HEADER_LIST_RE = re.compile(
    r'^\s*[-*]\s*\*\*[^*]+\*\*\s*[:—–-]',
    re.MULTILINE,
)

# "-ing" superficial analysis tacking
ING_TACKING_RE = re.compile(
    r',\s+(highlighting|emphasizing|showcasing|reflecting|demonstrating|'
    r'underscoring|illustrating|symbolizing|representing|signaling)\b',
    re.IGNORECASE,
)

# Formulaic "challenges and future" sections
CHALLENGES_FUTURE_RE = re.compile(
    r'\b(challenges?\s+and\s+(?:future\s+)?(?:prospects?|directions?|outlook|opportunities))\b',
    re.IGNORECASE,
)

# Generic positive conclusions
GENERIC_POSITIVE_RE = re.compile(
    r'\b(the future looks bright|exciting times? (?:lie|lay) ahead|'
    r'poised for (?:growth|success|greatness)|'
    r'paving the way for|'
    r'ushering in a new era)\b',
    re.IGNORECASE,
)


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def detect_structural_patterns(text: str, min_severity: Severity = Severity.LOW) -> list[PatternMatch]:
    """Detect structural AI writing patterns."""
    matches: list[PatternMatch] = []
    sentences = _split_sentences(text)

    # ── Connector density (Rosenfeld 2024) ──────────────────────────
    if len(sentences) >= 3:
        connector_count = 0
        text_lower = text.lower()
        for conn in DISCOURSE_CONNECTORS:
            connector_count += len(re.findall(
                rf'\b{re.escape(conn)}\b', text_lower
            ))

        # Per paragraph: ≥3 connectors in a block of sentences
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            para_lower = para.lower()
            para_sentences = _split_sentences(para)
            if len(para_sentences) < 3:
                continue
            para_conn = sum(
                1 for conn in DISCOURSE_CONNECTORS
                if re.search(rf'\b{re.escape(conn)}\b', para_lower)
            )
            if para_conn >= 3:
                start = text.find(para)
                matches.append(PatternMatch(
                    pattern_id="struct_connector_density",
                    category=PatternCategory.STRUCTURAL,
                    severity=Severity.HIGH,
                    start=start,
                    end=start + len(para),
                    matched_text=f"[{para_conn} connectors in paragraph]",
                    message=f"≥3 discourse connectors in one paragraph ({para_conn} found) — reads as machine-generated transitions",
                    source="Rosenfeld 2024",
                ))

    # ── Sentence length uniformity (Rosenfeld 2024) ─────────────────
    if len(sentences) >= 5:
        lengths = [len(s.split()) for s in sentences]
        mean_len = statistics.mean(lengths)
        std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0

        if std_len < 3.0 and mean_len > 5:
            matches.append(PatternMatch(
                pattern_id="struct_sentence_uniformity",
                category=PatternCategory.STRUCTURAL,
                severity=Severity.HIGH,
                start=0,
                end=len(text),
                matched_text=f"[stddev={std_len:.1f}, mean={mean_len:.1f}]",
                message=f"Sentence lengths too uniform (σ={std_len:.1f}) — human writing has σ>5",
                source="Rosenfeld 2024",
            ))

    # ── Rule of three overuse (Wikipedia Signs of AI) ───────────────
    r3_matches = list(RULE_OF_THREE_RE.finditer(text))
    if len(r3_matches) >= 3:
        matches.append(PatternMatch(
            pattern_id="struct_rule_of_three",
            category=PatternCategory.STRUCTURAL,
            severity=Severity.MEDIUM,
            start=r3_matches[0].start(),
            end=r3_matches[-1].end(),
            matched_text=f"[{len(r3_matches)} instances]",
            message=f"Rule-of-three pattern used {len(r3_matches)}× — AI forces ideas into triads",
            source="Wikipedia AI Signs",
        ))

    # ── Negative parallelism ────────────────────────────────────────
    for m in NEGATIVE_PARALLELISM_RE.finditer(text):
        matches.append(PatternMatch(
            pattern_id="struct_negative_parallelism",
            category=PatternCategory.STRUCTURAL,
            severity=Severity.MEDIUM,
            start=m.start(),
            end=m.end(),
            matched_text=m.group(),
            message="Negative parallelism (Not only...but also) — AI overuses this structure",
            source="Wikipedia AI Signs",
        ))

    # ── Em dash overuse ─────────────────────────────────────────────
    em_count = len(EM_DASH_RE.findall(text))
    if em_count >= 3:
        matches.append(PatternMatch(
            pattern_id="struct_em_dash_overuse",
            category=PatternCategory.STYLISTIC,
            severity=Severity.MEDIUM,
            start=0,
            end=len(text),
            matched_text=f"[{em_count} em dashes]",
            message=f"Em dash overuse ({em_count}×) — AI text uses em dashes more than humans",
            source="Wikipedia AI Signs",
        ))

    # ── Inline-header vertical lists ────────────────────────────────
    header_list_matches = list(INLINE_HEADER_LIST_RE.finditer(text))
    if len(header_list_matches) >= 3:
        matches.append(PatternMatch(
            pattern_id="struct_inline_header_list",
            category=PatternCategory.STYLISTIC,
            severity=Severity.MEDIUM,
            start=header_list_matches[0].start(),
            end=header_list_matches[-1].end(),
            matched_text=f"[{len(header_list_matches)} inline-header items]",
            message="Bolded inline-header list pattern — AI formatting tell",
            source="Wikipedia AI Signs",
        ))

    # ── Superficial -ing analysis tacking ───────────────────────────
    for m in ING_TACKING_RE.finditer(text):
        matches.append(PatternMatch(
            pattern_id="struct_ing_tacking",
            category=PatternCategory.STRUCTURAL,
            severity=Severity.HIGH,
            start=m.start(),
            end=m.end(),
            matched_text=m.group(),
            message="Superficial -ing tacking for fake depth",
            source="Wikipedia AI Signs",
        ))

    # ── Formulaic "challenges and future prospects" ─────────────────
    for m in CHALLENGES_FUTURE_RE.finditer(text):
        matches.append(PatternMatch(
            pattern_id="struct_challenges_future",
            category=PatternCategory.STRUCTURAL,
            severity=Severity.MEDIUM,
            start=m.start(),
            end=m.end(),
            matched_text=m.group(),
            message="Formulaic AI section structure",
            source="Wikipedia AI Signs",
        ))

    # ── Generic positive conclusions ────────────────────────────────
    for m in GENERIC_POSITIVE_RE.finditer(text):
        matches.append(PatternMatch(
            pattern_id="struct_generic_positive",
            category=PatternCategory.STRUCTURAL,
            severity=Severity.HIGH,
            start=m.start(),
            end=m.end(),
            matched_text=m.group(),
            message="Generic positive conclusion — strong AI tell",
            source="Wikipedia AI Signs",
        ))

    return matches
