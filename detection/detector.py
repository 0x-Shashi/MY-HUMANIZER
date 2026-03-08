"""
Unified AI pattern detection engine — combines all detectors.
Pass 1 of the pipeline: DETECT.
"""

from __future__ import annotations

from core.types import DetectionResult, Severity

from .patterns.communication import detect_communication_patterns
from .patterns.lexical import detect_lexical_patterns
from .patterns.structural import detect_structural_patterns
from .statistical_analyzer import compute_ai_score, compute_statistical_profile


def detect(text: str, min_severity: Severity = Severity.LOW) -> DetectionResult:
    """
    Run full AI detection analysis on input text.

    Combines:
    1. Lexical patterns (137+ word/phrase tells from corpus studies)
    2. Structural patterns (sentence uniformity, connector density, etc.)
    3. Communication patterns (chatbot artifacts, sycophantic tone)
    4. Statistical profiling (perplexity, burstiness, entropy, TTR)

    Returns a DetectionResult with patterns, stats, and composite AI score.
    """
    # Run all detectors
    lexical = detect_lexical_patterns(text, min_severity)
    structural = detect_structural_patterns(text, min_severity)
    communication = detect_communication_patterns(text, min_severity)

    all_patterns = lexical + structural + communication
    all_patterns.sort(key=lambda p: p.start)

    # Statistical profiling
    stats = compute_statistical_profile(text)

    # Composite AI score
    ai_score, breakdown = compute_ai_score(stats, len(all_patterns))

    return DetectionResult(
        patterns=all_patterns,
        stats=stats,
        ai_score=ai_score,
        breakdown=breakdown,
    )
