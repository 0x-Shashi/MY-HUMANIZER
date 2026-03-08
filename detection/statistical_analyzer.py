"""
Statistical text analyzer — computes perplexity, burstiness, entropy, and
other statistical fingerprints that AI detectors actually use.

This is the core differentiator: existing humanizers only do lexical patterns,
but real detectors (GPTZero, Originality.ai, Turnitin) use these metrics.

Sources:
- Rosenfeld & Lazebnik (2024): Sentence length variation, structural signals
- GPTZero whitepaper: Perplexity + burstiness as dual detection axes
- Mitchell et al. (2023): Log-likelihood curvature for detection
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter

from core.types import StatisticalProfile


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace+punctuation tokenizer."""
    return re.findall(r"\b\w+(?:'\w+)?\b", text.lower())


def compute_statistical_profile(text: str) -> StatisticalProfile:
    """
    Compute the full statistical fingerprint of a text.

    Human writing characteristics (Rosenfeld 2024):
    - Burstiness (sentence length σ): 6-10 for humans, 2-4 for AI
    - Vocabulary richness (TTR): 0.65-0.80 for humans, 0.50-0.65 for AI
    - Connector density: <0.15 for humans, >0.20 for AI
    - Perplexity: 50-150 for humans, 10-40 for AI (model-dependent)
    """
    sentences = _split_sentences(text)
    tokens = _tokenize(text)

    if not tokens or not sentences:
        return StatisticalProfile()

    # ── Sentence length statistics ──────────────────────────────────
    sent_lengths = [len(_tokenize(s)) for s in sentences]
    mean_sent_len = statistics.mean(sent_lengths)
    sent_len_var = statistics.variance(sent_lengths) if len(sent_lengths) > 1 else 0.0
    burstiness = statistics.stdev(sent_lengths) if len(sent_lengths) > 1 else 0.0

    # ── Vocabulary richness (Type-Token Ratio) ──────────────────────
    unique_tokens = set(tokens)
    ttr = len(unique_tokens) / len(tokens) if tokens else 0.0

    # ── Shannon entropy of word distribution ────────────────────────
    word_counts = Counter(tokens)
    total = len(tokens)
    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in word_counts.values()
        if count > 0
    )

    # ── Connector density ───────────────────────────────────────────
    connectors = {
        "moreover", "furthermore", "additionally", "consequently",
        "nevertheless", "nonetheless", "however", "therefore",
        "thus", "hence", "meanwhile", "subsequently", "alternatively",
        "conversely", "similarly", "likewise", "in addition",
    }
    text_lower = text.lower()
    connector_count = sum(
        1 for c in connectors
        if re.search(rf'\b{re.escape(c)}\b', text_lower)
    )
    connector_density = connector_count / len(sentences) if sentences else 0.0

    # ── Repetition score ────────────────────────────────────────────
    # Words appearing 3+ times, weighted by frequency
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "and",
        "but", "or", "nor", "not", "so", "yet", "both", "either",
        "neither", "each", "every", "all", "any", "few", "more",
        "most", "other", "some", "such", "no", "only", "own", "same",
        "than", "too", "very", "just", "because", "if", "when", "that",
        "this", "it", "its", "i", "you", "he", "she", "we", "they",
        "me", "him", "her", "us", "them", "my", "your", "his", "our",
        "their", "what", "which", "who", "whom", "whose", "where",
        "how", "about", "up", "out", "then", "there", "here",
    }
    content_counts = {
        w: c for w, c in word_counts.items()
        if w not in stop_words and c >= 3
    }
    repetition_score = (
        sum(content_counts.values()) / total if content_counts else 0.0
    )

    # ── Approximate perplexity (unigram model) ──────────────────────
    # True perplexity requires a language model, but unigram perplexity
    # correlates and is useful as a baseline / fast proxy.
    # PP = 2^H where H is cross-entropy
    perplexity = 2.0 ** entropy if entropy > 0 else 1.0

    # ── Average word frequency rank ─────────────────────────────────
    # How "common" the vocabulary is — AI uses higher-frequency words
    avg_freq_rank = _estimate_avg_word_rank(tokens)

    return StatisticalProfile(
        perplexity=perplexity,
        burstiness=burstiness,
        entropy=entropy,
        mean_sentence_length=mean_sent_len,
        sentence_length_variance=sent_len_var,
        vocabulary_richness=ttr,
        connector_density=connector_density,
        repetition_score=repetition_score,
        avg_word_frequency_rank=avg_freq_rank,
    )


def _estimate_avg_word_rank(tokens: list[str]) -> float:
    """
    Estimate average word frequency rank from a rough frequency tier system.
    Lower rank = more common words (AI tends to use more common vocabulary).

    This is a fast heuristic; a real implementation would use word frequency lists.
    """
    # Tier 1: top 100 most common English words (rank ~50)
    tier1 = {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us",
    }
    # Tier 2: next 1000 words (rank ~500)
    # We'll just check length heuristic since we don't have a full list
    total_rank = 0.0
    for t in tokens:
        if t in tier1:
            total_rank += 50
        elif len(t) <= 4:
            total_rank += 200
        elif len(t) <= 7:
            total_rank += 1000
        else:
            total_rank += 3000  # less common, longer words

    return total_rank / len(tokens) if tokens else 0.0


def compute_ai_score(profile: StatisticalProfile, patterns_count: int) -> tuple[float, dict[str, float]]:
    """
    Compute composite AI probability score from statistical profile and pattern count.

    Returns (score 0-1, breakdown dict).

    The scoring weights patterns, statistical signals, and structural markers:
    - Pattern matches: 30% (lexical AI tells)
    - Burstiness: 25% (sentence length variance — strongest single signal)
    - Vocabulary richness: 15%
    - Connector density: 15%
    - Repetition: 10%
    - Sentence uniformity: 5%
    """
    breakdown: dict[str, float] = {}

    # Pattern score: sigmoid on count (saturates around 15 patterns)
    pattern_raw = min(patterns_count / 15.0, 1.0)
    breakdown["patterns"] = pattern_raw

    # Burstiness: humans σ ∈ [6,10], AI σ ∈ [2,4]
    # Score: 1.0 if σ<2, 0.0 if σ>8, linear between
    if profile.burstiness < 2.0:
        bust_score = 1.0
    elif profile.burstiness > 8.0:
        bust_score = 0.0
    else:
        bust_score = 1.0 - (profile.burstiness - 2.0) / 6.0
    breakdown["burstiness"] = bust_score

    # Vocabulary richness: humans TTR ∈ [0.65,0.80], AI TTR ∈ [0.50,0.65]
    if profile.vocabulary_richness < 0.50:
        vocab_score = 1.0
    elif profile.vocabulary_richness > 0.75:
        vocab_score = 0.0
    else:
        vocab_score = 1.0 - (profile.vocabulary_richness - 0.50) / 0.25
    breakdown["vocabulary"] = vocab_score

    # Connector density: humans <0.15, AI >0.20
    if profile.connector_density > 0.25:
        conn_score = 1.0
    elif profile.connector_density < 0.10:
        conn_score = 0.0
    else:
        conn_score = (profile.connector_density - 0.10) / 0.15
    breakdown["connectors"] = conn_score

    # Repetition: AI tends to repeat content words less (more synonym cycling)
    # or more (same connector/transition phrases)
    rep_score = min(profile.repetition_score * 5.0, 1.0)
    breakdown["repetition"] = rep_score

    # Sentence uniformity: low variance = AI
    if profile.sentence_length_variance < 10.0:
        uniform_score = 1.0 - profile.sentence_length_variance / 10.0
    else:
        uniform_score = 0.0
    breakdown["uniformity"] = uniform_score

    # Weighted composite
    ai_score = (
        0.30 * pattern_raw
        + 0.25 * bust_score
        + 0.15 * vocab_score
        + 0.15 * conn_score
        + 0.10 * rep_score
        + 0.05 * uniform_score
    )

    return min(max(ai_score, 0.0), 1.0), breakdown
