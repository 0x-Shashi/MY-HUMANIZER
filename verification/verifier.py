"""
Verification engine — runs detection on transformed output and decides
whether another iteration is needed.

The key insight: existing humanizers do ONE pass and hope for the best.
We verify the output against the same metrics that real AI detectors use,
and iterate until the text actually passes.

Verification checks:
1. AI score below threshold (target: <0.30)
2. No CRITICAL/HIGH patterns remaining
3. Statistical profile in human range (burstiness, perplexity, TTR)
4. Meaning preserved (semantic similarity > threshold)
5. No new AI patterns introduced by the transformation itself
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.types import (
    DetectionResult,
    HumanizationConfig,
    Severity,
    StatisticalProfile,
)
from detection.detector import detect


@dataclass
class VerificationResult:
    """Result of verifying transformed text."""
    passed: bool = False
    ai_score: float = 1.0
    detection: DetectionResult = field(default_factory=DetectionResult)
    issues: list[str] = field(default_factory=list)
    stats_in_range: bool = False
    critical_patterns_remaining: int = 0
    high_patterns_remaining: int = 0
    semantic_similarity: float = 1.0


def verify(
    original_text: str,
    transformed_text: str,
    config: HumanizationConfig,
    previous_detection: DetectionResult | None = None,
) -> VerificationResult:
    """
    Verify that the transformation actually improved the text.

    Args:
        original_text: The original (AI-generated) input
        transformed_text: The transformed output
        config: Humanization configuration with targets
        previous_detection: Detection result from before transformation

    Returns:
        VerificationResult with pass/fail and diagnostics
    """
    result = VerificationResult()
    result.detection = detect(transformed_text, config.min_severity)
    result.ai_score = result.detection.ai_score

    # ── Check 1: AI score threshold ─────────────────────────────────
    ai_score_threshold = 0.22  # aggressive: target near-zero detection
    if result.ai_score > ai_score_threshold:
        result.issues.append(
            f"AI score {result.ai_score:.2f} exceeds threshold {ai_score_threshold}"
        )

    # ── Check 2: Critical/High patterns remaining ───────────────────
    severity_counts = result.detection.pattern_count_by_severity
    result.critical_patterns_remaining = severity_counts.get(Severity.CRITICAL, 0)
    result.high_patterns_remaining = severity_counts.get(Severity.HIGH, 0)

    if result.critical_patterns_remaining > 0:
        result.issues.append(
            f"{result.critical_patterns_remaining} CRITICAL patterns still present"
        )
    if result.high_patterns_remaining > 2:
        result.issues.append(
            f"{result.high_patterns_remaining} HIGH patterns still present (max 2)"
        )

    # ── Check 3: Statistical profile in human range ─────────────────
    stats = result.detection.stats
    result.stats_in_range = _check_stats_in_range(stats, config)
    if not result.stats_in_range:
        stat_issues = _get_stat_issues(stats, config)
        result.issues.extend(stat_issues)

    # ── Check 4: Meaning preservation ───────────────────────────────
    result.semantic_similarity = _compute_semantic_similarity(
        original_text, transformed_text
    )
    min_similarity = config.preserve_meaning * 0.9  # scaled threshold
    if result.semantic_similarity < min_similarity:
        result.issues.append(
            f"Semantic similarity {result.semantic_similarity:.2f} "
            f"below threshold {min_similarity:.2f}"
        )

    # ── Check 5: No new patterns introduced ─────────────────────────
    if previous_detection is not None:
        new_patterns = _find_new_patterns(previous_detection, result.detection)
        if new_patterns:
            result.issues.append(
                f"{len(new_patterns)} new AI patterns introduced: "
                + ", ".join(new_patterns[:3])
            )

    # ── Final verdict ───────────────────────────────────────────────
    result.passed = len(result.issues) == 0

    return result


def _check_stats_in_range(
    stats: StatisticalProfile, config: HumanizationConfig
) -> bool:
    """Check if all statistical metrics are in the human range."""
    # Burstiness: target ± 3
    if abs(stats.burstiness - config.target_burstiness) > 3.0:
        return False

    # Vocabulary richness: target ± 0.10
    if abs(stats.vocabulary_richness - config.target_vocabulary_richness) > 0.10:
        return False

    # Connector density: must be below max
    if stats.connector_density > config.target_connector_density_max:
        return False

    return True


def _get_stat_issues(
    stats: StatisticalProfile, config: HumanizationConfig
) -> list[str]:
    """Get specific stat issues for diagnostics."""
    issues = []

    if stats.burstiness < config.target_burstiness - 3.0:
        issues.append(
            f"Burstiness {stats.burstiness:.1f} too low "
            f"(target: {config.target_burstiness:.1f})"
        )
    elif stats.burstiness > config.target_burstiness + 3.0:
        issues.append(
            f"Burstiness {stats.burstiness:.1f} too high "
            f"(target: {config.target_burstiness:.1f})"
        )

    if stats.vocabulary_richness < config.target_vocabulary_richness - 0.10:
        issues.append(
            f"Vocabulary richness {stats.vocabulary_richness:.2f} too low "
            f"(target: {config.target_vocabulary_richness:.2f})"
        )

    if stats.connector_density > config.target_connector_density_max:
        issues.append(
            f"Connector density {stats.connector_density:.2f} too high "
            f"(max: {config.target_connector_density_max:.2f})"
        )

    return issues


def _compute_semantic_similarity(original: str, transformed: str) -> float:
    """
    Compute semantic similarity between original and transformed text.

    Uses two approaches:
    1. Advanced: sentence-transformers cosine similarity (if available)
    2. Fallback: word overlap ratio (Jaccard + weighted)
    """
    try:
        return _semantic_similarity_dense(original, transformed)
    except (ImportError, Exception):
        return _semantic_similarity_overlap(original, transformed)


def _semantic_similarity_dense(original: str, transformed: str) -> float:
    """Compute cosine similarity using sentence-transformers."""
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([original, transformed], normalize_embeddings=True)
    similarity = float(np.dot(embeddings[0], embeddings[1]))
    return max(0.0, min(1.0, similarity))


def _semantic_similarity_overlap(original: str, transformed: str) -> float:
    """Fallback: weighted word overlap similarity."""
    import re

    def _tokens(text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    orig_tokens = _tokens(original)
    trans_tokens = _tokens(transformed)

    if not orig_tokens or not trans_tokens:
        return 0.0

    orig_set = set(orig_tokens)
    trans_set = set(trans_tokens)

    # Jaccard similarity
    intersection = orig_set & trans_set
    union = orig_set | trans_set
    jaccard = len(intersection) / len(union) if union else 0.0

    # Order-aware: how many words stayed in roughly the same position
    # (higher = more preserved structure)
    min_len = min(len(orig_tokens), len(trans_tokens))
    positional_matches = sum(
        1 for i in range(min_len)
        if orig_tokens[i] == trans_tokens[i]
    )
    positional_score = positional_matches / min_len if min_len else 0.0

    # Length ratio (penalize significant length changes)
    length_ratio = min(len(orig_tokens), len(trans_tokens)) / max(
        len(orig_tokens), len(trans_tokens)
    )

    # Weighted combination
    return 0.5 * jaccard + 0.3 * positional_score + 0.2 * length_ratio


def _find_new_patterns(
    before: DetectionResult, after: DetectionResult
) -> list[str]:
    """Find AI patterns that were introduced by the transformation."""
    before_ids = {p.pattern_id for p in before.patterns}
    new_patterns = []
    for p in after.patterns:
        if p.pattern_id not in before_ids:
            new_patterns.append(p.pattern_id)
    return new_patterns


def get_iteration_guidance(result: VerificationResult) -> dict[str, bool]:
    """
    Based on verification failures, determine which transformation passes
    should be re-run in the next iteration.

    Returns dict of pass names → should_run.
    """
    guidance = {
        "word_level": False,
        "sentence_level": False,
        "burstiness": False,
        "voice": False,
    }

    for issue in result.issues:
        issue_lower = issue.lower()

        if "critical patterns" in issue_lower or "high patterns" in issue_lower:
            guidance["word_level"] = True

        if "burstiness" in issue_lower:
            guidance["burstiness"] = True

        if "connector density" in issue_lower:
            guidance["sentence_level"] = True

        if "vocabulary richness" in issue_lower:
            guidance["word_level"] = True

        if "semantic similarity" in issue_lower:
            # Reduce aggression, don't redo voice injection
            guidance["voice"] = False

        if "new ai patterns" in issue_lower:
            guidance["word_level"] = True

        if "ai score" in issue_lower:
            guidance["word_level"] = True
            guidance["sentence_level"] = True

    # If AI score is still high and no specific issue flagged, run all
    if result.ai_score > 0.30 and not any(guidance.values()):
        guidance["word_level"] = True
        guidance["sentence_level"] = True
        guidance["burstiness"] = True

    return guidance
