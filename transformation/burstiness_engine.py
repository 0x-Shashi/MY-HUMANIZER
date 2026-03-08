"""
Burstiness engine & perplexity shaper.

THE critical differentiator: what detectors ACTUALLY measure.

GPTZero's dual axes:
1. Perplexity: How "surprised" a language model is by the text
   - AI text: low perplexity (10-40) — predictable word choices
   - Human text: higher perplexity (50-150) — surprising word choices
2. Burstiness: Variation in sentence complexity
   - AI text: low burstiness (σ=2-4) — uniform sentence lengths
   - Human text: high burstiness (σ=6-10) — mix of short punchy + long complex

This engine shapes BOTH metrics to match human distributions by:
- Inserting calculated "surprise" words (controlled perplexity injection)
- Forcing sentence length variance (short-long-short-long pattern)
- Adding sentence fragments and very short sentences
- Varying clause complexity within paragraphs
"""

from __future__ import annotations

import math
import random
import re
import statistics
from typing import Optional

from core.types import HumanizationConfig, TransformationChange, TransformPass, WritingDomain
from utils.nlp import Sentence, split_sentences


# Short sentence fragments humans naturally insert — domain-specific
FRAGMENT_INSERTIONS: dict[WritingDomain, list[str]] = {
    WritingDomain.CASUAL: [
        "Makes sense.",
        "Not always, though.",
        "Fair point.",
        "Big difference.",
        "Hard to say.",
        "Depends on context.",
        "Not exactly.",
        "Sort of.",
        "One catch, though.",
        "That said.",
        "In theory, at least.",
    ],
    WritingDomain.ACADEMIC: [
        "This matters.",
        "A key distinction.",
        "Not universally, though.",
        "Context matters here.",
        "The evidence is mixed.",
        "An open question.",
        "The data supports this.",
        "A common misconception.",
        "The reverse also holds.",
        "This is debatable.",
        "Not without caveats.",
        "Results vary.",
    ],
    WritingDomain.TECHNICAL: [
        "Worth noting.",
        "Edge case here.",
        "Depends on the implementation.",
        "Not always true.",
        "Tricky part here.",
        "Same idea, different angle.",
        "A key distinction.",
        "One catch, though.",
        "In practice, at least.",
    ],
    WritingDomain.CREATIVE: [
        "Funny thing, that.",
        "Makes you think.",
        "Not always, though.",
        "Something to consider.",
        "Hard to pin down.",
        "Fair point.",
        "In a way.",
    ],
    WritingDomain.BUSINESS: [
        "Big difference.",
        "Worth keeping in mind.",
        "Depends on context.",
        "Not always the case.",
        "A key takeaway.",
        "That said.",
        "The data backs this up.",
    ],
}

# Words that increase perplexity (less predictable vocabulary)
# These are real English words that language models assign lower probability
SURPRISE_MODIFIERS = {
    "actually": ["honestly", "frankly", "truthfully"],
    "very": ["remarkably", "strikingly", "unusually"],
    "important": ["non-trivial", "consequential", "weighty"],
    "good": ["decent", "respectable", "solid"],
    "bad": ["problematic", "rough", "messy"],
    "big": ["sizable", "substantial", "non-trivial"],
    "small": ["modest", "marginal", "slight"],
    "many": ["a handful of", "quite a few", "a number of"],
    "some": ["a few", "certain", "particular"],
    "often": ["time and again", "routinely", "frequently enough"],
}

# Sentence expansion templates (for making short sentences longer)
EXPANSION_TEMPLATES = [
    "To put it differently, {sentence}",
    "What this means in practice is that {sentence}",
    "{sentence} — though the details get complicated.",
    "{sentence}, which is something worth keeping in mind.",
]

# Sentence compression patterns (for shortening long sentences)
COMPRESSION_REMOVE_PHRASES = [
    "it is important to note that ",
    "it should be mentioned that ",
    "it is worth noting that ",
    "it is clear that ",
    "it seems evident that ",
    "it goes without saying that ",
    "needless to say, ",
]


def shape_burstiness(
    text: str,
    config: HumanizationConfig,
    rng: Optional[random.Random] = None,
) -> tuple[str, list[TransformationChange]]:
    """
    Shape sentence-length distribution to match human burstiness patterns.

    Target: σ > 5.0 (human range 6-10), with natural-looking variation.
    Strategy: Create a rhythm of short-medium-long-short-long sentences.
    """
    if rng is None:
        rng = random.Random()

    changes: list[TransformationChange] = []
    sentences = split_sentences(text)

    if len(sentences) < 4:
        return text, changes

    # Measure current burstiness
    lengths = [s.word_count for s in sentences]
    current_std = statistics.stdev(lengths) if len(lengths) > 1 else 0
    target_std = config.target_sentence_length_std

    if current_std >= target_std * 0.85:
        return text, changes  # Already bursty enough

    # Strategy: We need to INCREASE variance
    # 1. Make some sentences shorter (fragments, compression)
    # 2. Make some sentences longer (expansion, clause addition)
    # 3. Insert 1-3 word fragment sentences between longer ones

    result_sentences: list[str] = []
    modified = False
    used_fragments: set[str] = set()

    for i, s in enumerate(sentences):
        # Every ~4th sentence, insert a short fragment BEFORE it
        if (
            i > 0 and i % 4 == 0
            and rng.random() < 0.4 + 0.3 * config.creativity
            and current_std < target_std * 0.7
        ):
            domain_fragments = FRAGMENT_INSERTIONS.get(
                config.domain, FRAGMENT_INSERTIONS[WritingDomain.CASUAL]
            )
            available = [f for f in domain_fragments if f not in used_fragments]
            if not available:
                available = domain_fragments
            fragment = rng.choice(available)
            used_fragments.add(fragment)
            result_sentences.append(fragment)
            changes.append(TransformationChange(
                original="",
                replacement=fragment,
                start=s.start,
                end=s.start,
                pass_name=TransformPass.INJECT,
                reason="Fragment insertion for burstiness",
                confidence=0.7,
            ))
            modified = True

        # Shorten sentences that are close to the mean (reduce uniformity)
        if (
            abs(s.word_count - statistics.mean(lengths)) < 3
            and s.word_count > 10
            and rng.random() < 0.3
        ):
            shortened = _compress_sentence(s.text, rng)
            if shortened != s.text:
                result_sentences.append(shortened)
                changes.append(TransformationChange(
                    original=s.text,
                    replacement=shortened,
                    start=s.start,
                    end=s.end,
                    pass_name=TransformPass.INJECT,
                    reason="Sentence compression for burstiness",
                    confidence=0.7,
                ))
                modified = True
                continue

        result_sentences.append(s.text)

    if not modified:
        return text, changes

    new_text = " ".join(result_sentences)
    return new_text, changes


def inject_perplexity(
    text: str,
    config: HumanizationConfig,
    rng: Optional[random.Random] = None,
) -> tuple[str, list[TransformationChange]]:
    """
    Inject controlled "surprise" into text to raise perplexity.

    AI text uses the most probable next word; humans occasionally
    pick less obvious words. We selectively replace common words
    with less-predictable synonyms.

    This is NOT synonym replacement for detection evasion —
    it's specifically targeting the perplexity metric.
    """
    if rng is None:
        rng = random.Random()

    changes: list[TransformationChange] = []

    # Only inject in ~15-25% of opportunities
    inject_rate = 0.15 + 0.10 * config.creativity

    for common_word, replacements in SURPRISE_MODIFIERS.items():
        pattern = re.compile(rf'\b{re.escape(common_word)}\b', re.IGNORECASE)
        matches = list(pattern.finditer(text))

        for match in reversed(matches):
            if rng.random() > inject_rate:
                continue

            original = match.group()
            replacement = rng.choice(replacements)
            replacement = _preserve_case(original, replacement)

            text = text[:match.start()] + replacement + text[match.end():]
            changes.append(TransformationChange(
                original=original,
                replacement=replacement,
                start=match.start(),
                end=match.end(),
                pass_name=TransformPass.INJECT,
                reason="Perplexity injection — less predictable word choice",
                confidence=0.6,
            ))

    return text, changes


def _compress_sentence(text: str, rng: random.Random) -> str:
    """Try to compress a sentence by removing filler phrases."""
    result = text.lower()
    for phrase in COMPRESSION_REMOVE_PHRASES:
        if phrase in result:
            idx = result.find(phrase)
            original_phrase = text[idx:idx + len(phrase)]
            text = text[:idx] + text[idx + len(original_phrase):]
            if text and text[0].islower():
                text = text[0].upper() + text[1:]
            return text
    return text


def _preserve_case(original: str, replacement: str) -> str:
    """Transfer casing from original to replacement."""
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement
