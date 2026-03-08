"""
Sentence-level transformation — restructuring, splitting, merging, reordering.

This tackles the loophole that ALL existing humanizers miss:
they work word-by-word but never change sentence STRUCTURE.

AI detectors look at:
- Sentence length distribution (AI clusters around 15-25 words)
- Syntactic monotony (AI uses similar clause structures)
- Transition word density per paragraph

Techniques:
1. Sentence splitting (long → two shorter)
2. Sentence merging (two short → one compound)
3. Clause reordering (move dependent clause to front/back)
4. Transition reduction (remove excess connectors)
5. Parenthetical insertion (add human-like asides)
"""

from __future__ import annotations

import random
import re
from typing import Optional

from core.types import (
    HumanizationConfig,
    TransformationChange,
    TransformPass,
)
from utils.nlp import Sentence, split_sentences


# Coordinating conjunctions that can be split points
SPLIT_CONJUNCTIONS = {"and", "but", "yet", "so", "while", "whereas", "although"}

# Transition words to potentially remove (reduce connector density)
REMOVABLE_TRANSITIONS = {
    "moreover", "furthermore", "additionally", "consequently",
    "in addition", "as a result", "in conclusion", "therefore",
    "hence", "thus", "subsequently", "alternatively",
    "conversely", "similarly", "likewise", "nevertheless",
    "nonetheless", "meanwhile", "in summary", "to summarize",
    "in other words", "that being said", "it is worth noting that",
    "it is important to note that", "it should be noted that",
}

# Human-like parenthetical insertions
PARENTHETICALS = [
    " — and this matters —",
    " — at least in theory —",
    " — to some extent —",
    " (in practice)",
    " (roughly speaking)",
    " (as you'd expect)",
    ", of course,",
    ", in a sense,",
    ", to be fair,",
    ", admittedly,",
    ", surprisingly,",
    ", interestingly,",
    ", for better or worse,",
]

# Sentence starters that vary rhythm
SHORT_STARTERS = [
    "Still.", "True.", "Fair enough.", "Point taken.",
    "Right.", "Exactly.", "Not quite.", "Perhaps.",
]

# Casual hedges humans use
HUMAN_HEDGES = [
    "I think", "I'd argue", "It seems like", "My sense is that",
    "The way I see it,", "If I had to guess,", "In my experience,",
    "From what I can tell,",
]


def transform_sentences(
    text: str,
    config: HumanizationConfig,
    rng: Optional[random.Random] = None,
) -> tuple[str, list[TransformationChange]]:
    """
    Apply sentence-level transformations to improve structure variety.
    """
    if rng is None:
        rng = random.Random()

    changes: list[TransformationChange] = []
    sentences = split_sentences(text)

    if len(sentences) < 2:
        return text, changes

    # Phase 1: Reduce connector density
    result_sentences, conn_changes = _reduce_connectors(sentences, config, rng)
    changes.extend(conn_changes)

    # Phase 2: Split overly long sentences
    result_sentences, split_changes = _split_long_sentences(result_sentences, config, rng)
    changes.extend(split_changes)

    # Phase 3: Merge overly short adjacent sentences
    result_sentences, merge_changes = _merge_short_sentences(result_sentences, config, rng)
    changes.extend(merge_changes)

    # Phase 4: Reorder clauses in some sentences
    result_sentences, reorder_changes = _reorder_clauses(result_sentences, config, rng)
    changes.extend(reorder_changes)

    # Phase 5: Insert occasional parenthetical asides (personality)
    result_sentences, paren_changes = _insert_parentheticals(result_sentences, config, rng)
    changes.extend(paren_changes)

    # Reconstruct text
    new_text = " ".join(s.text for s in result_sentences)
    return new_text, changes


def _reduce_connectors(
    sentences: list[Sentence],
    config: HumanizationConfig,
    rng: random.Random,
) -> tuple[list[Sentence], list[TransformationChange]]:
    """Remove excess discourse connectors from sentence starts."""
    changes: list[TransformationChange] = []
    result = []

    connector_count = 0
    for s in sentences:
        first_word_match = re.match(r'^(\w+)', s.text.lower())
        first_word = first_word_match.group(1) if first_word_match else ""

        # Check for multi-word transitions too
        starts_with_connector = False
        matched_connector = ""
        for trans in REMOVABLE_TRANSITIONS:
            if s.text.lower().startswith(trans):
                starts_with_connector = True
                matched_connector = trans
                connector_count += 1
                break

        if not starts_with_connector and first_word in REMOVABLE_TRANSITIONS:
            starts_with_connector = True
            matched_connector = first_word
            connector_count += 1

        # Remove if we've seen too many (any after the first)
        if starts_with_connector and connector_count > 0 and rng.random() < 0.95:
            # Strip the connector from the sentence start
            new_text = s.text[len(matched_connector):].lstrip(" ,;")
            if new_text:
                new_text = new_text[0].upper() + new_text[1:]
                changes.append(TransformationChange(
                    original=s.text,
                    replacement=new_text,
                    start=s.start,
                    end=s.end,
                    pass_name=TransformPass.REWRITE,
                    reason="Reduce connector density (AI tell)",
                    confidence=0.85,
                ))
                result.append(Sentence(
                    text=new_text,
                    start=s.start,
                    end=s.end,
                    word_count=s.word_count - 1,
                ))
                continue

        result.append(s)

    return result, changes


def _split_long_sentences(
    sentences: list[Sentence],
    config: HumanizationConfig,
    rng: random.Random,
) -> tuple[list[Sentence], list[TransformationChange]]:
    """Split sentences that are too long (>30 words) at conjunction points."""
    changes: list[TransformationChange] = []
    result: list[Sentence] = []

    for s in sentences:
        if s.word_count <= 30 or rng.random() > 0.6:
            result.append(s)
            continue

        # Find a split point at a conjunction
        split_point = _find_split_point(s.text)
        if split_point is None:
            result.append(s)
            continue

        first_half = s.text[:split_point].rstrip(" ,;")
        second_half = s.text[split_point:].lstrip(" ,;")

        # Clean up the conjunction if it's at the start of second half
        for conj in SPLIT_CONJUNCTIONS:
            if second_half.lower().startswith(conj):
                second_half = second_half[len(conj):].lstrip(" ,")
                break

        if not first_half.endswith((".", "!", "?")):
            first_half += "."
        if second_half:
            second_half = second_half[0].upper() + second_half[1:]

        if len(first_half.split()) >= 5 and len(second_half.split()) >= 5:
            changes.append(TransformationChange(
                original=s.text,
                replacement=f"{first_half} {second_half}",
                start=s.start,
                end=s.end,
                pass_name=TransformPass.REWRITE,
                reason="Split long sentence for burstiness",
                confidence=0.8,
            ))
            result.append(Sentence(
                text=first_half, start=s.start,
                end=s.start + len(first_half),
                word_count=len(first_half.split()),
            ))
            result.append(Sentence(
                text=second_half,
                start=s.start + len(first_half) + 1,
                end=s.end,
                word_count=len(second_half.split()),
            ))
        else:
            result.append(s)

    return result, changes


def _merge_short_sentences(
    sentences: list[Sentence],
    config: HumanizationConfig,
    rng: random.Random,
) -> tuple[list[Sentence], list[TransformationChange]]:
    """Merge adjacent very short sentences (<7 words) into compound sentences."""
    changes: list[TransformationChange] = []
    result: list[Sentence] = []
    i = 0

    while i < len(sentences):
        if (
            i + 1 < len(sentences)
            and sentences[i].word_count < 7
            and sentences[i + 1].word_count < 7
            and rng.random() < 0.5
        ):
            # Merge with a conjunction
            conj = rng.choice([", and ", ", but ", " — ", "; "])
            first = sentences[i].text.rstrip(".")
            second = sentences[i + 1].text
            second_lower = second[0].lower() + second[1:] if second else ""
            merged = first + conj + second_lower

            changes.append(TransformationChange(
                original=f"{sentences[i].text} {sentences[i + 1].text}",
                replacement=merged,
                start=sentences[i].start,
                end=sentences[i + 1].end,
                pass_name=TransformPass.REWRITE,
                reason="Merge short sentences for variety",
                confidence=0.75,
            ))
            result.append(Sentence(
                text=merged,
                start=sentences[i].start,
                end=sentences[i + 1].end,
                word_count=sentences[i].word_count + sentences[i + 1].word_count,
            ))
            i += 2
        else:
            result.append(sentences[i])
            i += 1

    return result, changes


def _reorder_clauses(
    sentences: list[Sentence],
    config: HumanizationConfig,
    rng: random.Random,
) -> tuple[list[Sentence], list[TransformationChange]]:
    """Move dependent clauses from end to beginning or vice versa."""
    changes: list[TransformationChange] = []
    result: list[Sentence] = []

    subordinators = [
        "because", "although", "while", "since", "when",
        "if", "unless", "before", "after", "until",
    ]

    for s in sentences:
        if rng.random() > 0.25 * config.creativity:
            result.append(s)
            continue

        # Check for trailing dependent clause: ", because/although/while ..."
        # Only match proper dependent clauses (must have subject+verb, no internal commas)
        for sub in subordinators:
            pattern = re.compile(
                rf',\s+({re.escape(sub)}\s+[^,]+)$', re.IGNORECASE
            )
            match = pattern.search(s.text.rstrip(".!?"))
            if match and len(match.group(1).split()) >= 4:
                tail = s.text[len(s.text.rstrip(".!?")):]  # preserve terminal punctuation
                dep_clause = match.group(1)
                main_clause = s.text[:match.start()]

                # Move dependent clause to front
                reordered = f"{dep_clause[0].upper()}{dep_clause[1:]}, {main_clause[0].lower()}{main_clause[1:]}{tail}"
                changes.append(TransformationChange(
                    original=s.text,
                    replacement=reordered,
                    start=s.start,
                    end=s.end,
                    pass_name=TransformPass.REWRITE,
                    reason="Clause reorder for structural variety",
                    confidence=0.7,
                ))
                result.append(Sentence(
                    text=reordered, start=s.start, end=s.end,
                    word_count=s.word_count,
                ))
                break
        else:
            result.append(s)

    return result, changes


def _insert_parentheticals(
    sentences: list[Sentence],
    config: HumanizationConfig,
    rng: random.Random,
) -> tuple[list[Sentence], list[TransformationChange]]:
    """Insert human-like parenthetical asides into some sentences."""
    changes: list[TransformationChange] = []
    result: list[Sentence] = []

    # Only insert in 10-20% of sentences
    insert_prob = 0.10 + 0.10 * config.creativity

    for s in sentences:
        if s.word_count < 8 or rng.random() > insert_prob:
            result.append(s)
            continue

        # Skip if sentence already has a parenthetical or aside
        if any(marker in s.text for marker in (" — ", " (", ", of course,", ", admittedly,", ", surprisingly,", ", interestingly,")):
            result.append(s)
            continue

        # Find a good insertion point (after a comma or mid-sentence)
        words = s.text.split()
        mid = len(words) // 2
        insert_pos = None

        # Try to find a comma near the middle
        for i in range(max(3, mid - 3), min(len(words) - 2, mid + 3)):
            word = words[i]
            if word.endswith(","):
                insert_pos = i
                break

        if insert_pos is None:
            result.append(s)
            continue

        parenthetical = rng.choice(PARENTHETICALS)
        words[insert_pos] = words[insert_pos].rstrip(",") + parenthetical
        new_text = " ".join(words)

        changes.append(TransformationChange(
            original=s.text,
            replacement=new_text,
            start=s.start,
            end=s.end,
            pass_name=TransformPass.INJECT,
            reason="Parenthetical aside for human voice",
            confidence=0.65,
        ))
        result.append(Sentence(
            text=new_text, start=s.start, end=s.end,
            word_count=s.word_count + len(parenthetical.split()),
        ))

    return result, changes


def _find_split_point(text: str) -> Optional[int]:
    """Find the best point to split a sentence (at a conjunction)."""
    words = text.split()
    mid = len(words) // 2

    # Look for conjunctions near the middle
    best_pos = None
    best_dist = len(words)

    char_pos = 0
    for i, word in enumerate(words):
        if word.rstrip(",.;:") in SPLIT_CONJUNCTIONS and abs(i - mid) < best_dist:
            best_dist = abs(i - mid)
            best_pos = char_pos
        char_pos += len(word) + 1  # +1 for space

    return best_pos
