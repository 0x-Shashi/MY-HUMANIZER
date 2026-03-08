"""
Communication/style AI patterns — chatbot artifacts, tone, and formatting.

Sources:
- Juzek & Ward (COLING 2025): RLHF sycophantic patterns
- Wikipedia "Signs of AI writing": Formatting and communication artifacts
"""

from __future__ import annotations

import re

from core.types import PatternCategory, PatternMatch, Severity


# Vague attributions / weasel words
WEASEL_PATTERNS = [
    "experts argue", "experts suggest", "industry reports suggest",
    "observers have cited", "analysts believe", "many experts",
    "some researchers", "it has been argued", "studies show",
    "research suggests", "it is widely believed", "it is generally accepted",
    "some scholars argue",
]

# Knowledge cutoff disclaimers
CUTOFF_PATTERNS = [
    "as of my last", "as of my knowledge", "as of my training",
    "I don't have information beyond", "my training data",
    "as of my last update",
]

# Collaborative/chatbot artifacts
COLLABORATION_PATTERNS = [
    "Would you like me to", "Shall I", "Do you want me to",
    "Let me know if you'd like", "Is there anything else",
    "I can also", "If you need further", "Here's a",
    "Here is a", "Below is a",
]

# "False range" pattern: "from X to Y" where X and Y aren't on a scale
FALSE_RANGE_RE = re.compile(
    r'\bfrom\s+(\w+(?:\s+\w+){0,3})\s+to\s+(\w+(?:\s+\w+){0,3})\b',
    re.IGNORECASE,
)

# Elegant variation detection: tracking noun phrases and checking synonym cycling
# We do this by finding sentences with "also known as" / "referred to as" patterns
SYNONYM_CYCLING_RE = re.compile(
    r'\b(?:also known as|sometimes (?:called|referred to as)|'
    r'often (?:called|termed|described as)|alternatively (?:known|called))\b',
    re.IGNORECASE,
)

# Emoji in prose (not code)
EMOJI_RE = re.compile(
    r'[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]'
)

# Curly quotation marks (AI tends to produce these)
CURLY_QUOTE_RE = re.compile(r'[\u201C\u201D\u2018\u2019]')


def detect_communication_patterns(text: str, min_severity: Severity = Severity.LOW) -> list[PatternMatch]:
    """Detect communication/style AI patterns."""
    matches: list[PatternMatch] = []
    text_lower = text.lower()

    # ── Weasel words / vague attributions ───────────────────────────
    for pattern in WEASEL_PATTERNS:
        idx = text_lower.find(pattern.lower())
        if idx != -1:
            end = idx + len(pattern)
            matches.append(PatternMatch(
                pattern_id=f"comm_weasel_{pattern.replace(' ', '_')[:20]}",
                category=PatternCategory.COMMUNICATION,
                severity=Severity.MEDIUM,
                start=idx,
                end=end,
                matched_text=text[idx:end],
                message="Vague attribution — AI avoids citing specific sources",
                source="Wikipedia AI Signs",
            ))

    # ── Chatbot collaboration artifacts ─────────────────────────────
    for pattern in COLLABORATION_PATTERNS:
        idx = text_lower.find(pattern.lower())
        if idx != -1:
            end = idx + len(pattern)
            matches.append(PatternMatch(
                pattern_id=f"comm_collab_{pattern.replace(' ', '_')[:20]}",
                category=PatternCategory.COMMUNICATION,
                severity=Severity.CRITICAL,
                start=idx,
                end=end,
                matched_text=text[idx:end],
                message="Chatbot collaboration artifact left in text",
                source="Juzek 2025",
            ))

    # ── Knowledge cutoff disclaimers ────────────────────────────────
    for pattern in CUTOFF_PATTERNS:
        idx = text_lower.find(pattern.lower())
        if idx != -1:
            end = idx + len(pattern)
            matches.append(PatternMatch(
                pattern_id="comm_cutoff_disclaimer",
                category=PatternCategory.COMMUNICATION,
                severity=Severity.CRITICAL,
                start=idx,
                end=end,
                matched_text=text[idx:end],
                message="Knowledge cutoff disclaimer — dead giveaway of AI",
                source="Wikipedia AI Signs",
            ))

    # ── Emoji in prose ──────────────────────────────────────────────
    emoji_matches = list(EMOJI_RE.finditer(text))
    if emoji_matches:
        matches.append(PatternMatch(
            pattern_id="comm_emoji_in_prose",
            category=PatternCategory.STYLISTIC,
            severity=Severity.MEDIUM,
            start=emoji_matches[0].start(),
            end=emoji_matches[-1].end(),
            matched_text=f"[{len(emoji_matches)} emojis]",
            message="Emojis in prose — common AI formatting tell",
            source="Wikipedia AI Signs",
        ))

    # ── Superficial -ing analysis tacking (sentence-ending) ─────────
    # Already handled in structural, but catch sentence-initial ones
    ing_start_re = re.compile(
        r'(?:^|\.\s+)(Highlighting|Emphasizing|Showcasing|Reflecting|'
        r'Demonstrating|Underscoring|Illustrating|Symbolizing)\b',
        re.MULTILINE,
    )
    for m in ing_start_re.finditer(text):
        matches.append(PatternMatch(
            pattern_id="comm_ing_opener",
            category=PatternCategory.COMMUNICATION,
            severity=Severity.HIGH,
            start=m.start(),
            end=m.end(),
            matched_text=m.group(),
            message="Sentence starting with -ing participle for fake depth",
            source="Wikipedia AI Signs",
        ))

    return matches
