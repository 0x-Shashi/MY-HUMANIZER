"""
NLP utility functions — tokenization, sentence splitting, POS tagging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Sentence:
    """A sentence with metadata."""
    text: str
    start: int  # char offset in original text
    end: int
    word_count: int


def split_sentences(text: str) -> list[Sentence]:
    """
    Split text into sentences preserving offsets.
    Handles abbreviations, decimals, and common edge cases.
    """
    # First, protect known abbreviations from being split
    _ABBREVS = {"Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
                "vs.", "etc.", "e.g.", "i.e."}
    protected = text
    replacements: dict[str, str] = {}
    for abbr in _ABBREVS:
        placeholder = abbr.replace(".", "\x00")
        replacements[placeholder] = abbr
        protected = protected.replace(abbr, placeholder)

    # Sentence-ending punctuation followed by space + capital or end
    pattern = re.compile(
        r'(?<=[.!?])'         # after sentence-ending punctuation
        r'\s+'                # whitespace separator
        r'(?=[A-Z"\u201C])'   # before capital letter or opening quote
    )

    sentences: list[Sentence] = []
    last_end = 0

    for match in pattern.finditer(protected):
        sent_text = protected[last_end:match.start()].strip()
        if sent_text:
            # Restore abbreviations
            for placeholder, abbr in replacements.items():
                sent_text = sent_text.replace(placeholder, abbr)
            words = len(re.findall(r'\b\w+\b', sent_text))
            sentences.append(Sentence(
                text=sent_text,
                start=last_end,
                end=match.start(),
                word_count=words,
            ))
        last_end = match.end()

    # Last sentence
    remaining = protected[last_end:].strip()
    if remaining:
        # Restore abbreviations
        for placeholder, abbr in replacements.items():
            remaining = remaining.replace(placeholder, abbr)
        words = len(re.findall(r'\b\w+\b', remaining))
        sentences.append(Sentence(
            text=remaining,
            start=last_end,
            end=len(text),
            word_count=words,
        ))

    # Fallback: if splitting produced nothing, return whole text
    if not sentences and text.strip():
        words = len(re.findall(r'\b\w+\b', text))
        sentences.append(Sentence(text=text.strip(), start=0, end=len(text), word_count=words))

    return sentences


def tokenize_words(text: str) -> list[str]:
    """Simple word tokenizer that preserves contractions."""
    return re.findall(r"\b\w+(?:'\w+)?\b", text)


def get_word_spans(text: str) -> list[tuple[str, int, int]]:
    """Get words with their character offsets in the text."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\b\w+(?:'\w+)?\b", text)]
