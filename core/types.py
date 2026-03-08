"""
Core domain types for the My Humanizer engine.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class Severity(enum.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PatternCategory(enum.Enum):
    LEXICAL = "lexical"
    STRUCTURAL = "structural"
    STYLISTIC = "stylistic"
    COMMUNICATION = "communication"
    STATISTICAL = "statistical"


class WritingDomain(enum.Enum):
    ACADEMIC = "academic"
    CASUAL = "casual"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    BUSINESS = "business"


class TransformPass(enum.Enum):
    DETECT = "detect"
    RETRIEVE = "retrieve"
    REWRITE = "rewrite"
    INJECT = "inject"
    VERIFY = "verify"
    ITERATE = "iterate"


@dataclass
class PatternMatch:
    """A detected AI writing pattern in the text."""
    pattern_id: str
    category: PatternCategory
    severity: Severity
    start: int
    end: int
    matched_text: str
    replacement: Optional[str] = None
    message: str = ""
    source: str = ""  # research paper or empirical source


@dataclass
class StatisticalProfile:
    """Statistical fingerprint of a text block."""
    perplexity: float = 0.0
    burstiness: float = 0.0  # sentence length std dev
    entropy: float = 0.0
    mean_sentence_length: float = 0.0
    sentence_length_variance: float = 0.0
    vocabulary_richness: float = 0.0  # type-token ratio
    connector_density: float = 0.0
    repetition_score: float = 0.0
    avg_word_frequency_rank: float = 0.0

    @property
    def is_likely_ai(self) -> bool:
        """Heuristic: AI text has low burstiness + low perplexity."""
        return (
            self.burstiness < 4.0
            and self.sentence_length_variance < 15.0
        )


@dataclass
class DetectionResult:
    """Full detection analysis of input text."""
    patterns: list[PatternMatch] = field(default_factory=list)
    stats: StatisticalProfile = field(default_factory=StatisticalProfile)
    ai_score: float = 0.0  # 0 = definitely human, 1 = definitely AI
    breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def pattern_count_by_severity(self) -> dict[Severity, int]:
        counts: dict[Severity, int] = {}
        for p in self.patterns:
            counts[p.severity] = counts.get(p.severity, 0) + 1
        return counts


@dataclass
class TransformationChange:
    """A single change applied during transformation."""
    original: str
    replacement: str
    start: int
    end: int
    pass_name: TransformPass
    reason: str = ""
    confidence: float = 1.0


@dataclass
class TransformationResult:
    """Result of the full transformation pipeline."""
    original_text: str
    transformed_text: str
    changes: list[TransformationChange] = field(default_factory=list)
    detection_before: DetectionResult = field(default_factory=DetectionResult)
    detection_after: DetectionResult = field(default_factory=DetectionResult)
    iterations: int = 0
    domain: WritingDomain = WritingDomain.CASUAL

    @property
    def ai_score_reduction(self) -> float:
        return self.detection_before.ai_score - self.detection_after.ai_score

    @property
    def change_count(self) -> int:
        return len(self.changes)


@dataclass
class HumanizationConfig:
    """Configuration for a humanization run."""
    domain: WritingDomain = WritingDomain.ACADEMIC
    creativity: float = 0.7  # 0-1 (higher = more aggressive replacement)
    preserve_meaning: float = 0.75  # 0-1: how strictly to preserve original meaning
    target_burstiness: float = 7.5  # target sentence-length std dev
    target_perplexity_range: tuple[float, float] = (50.0, 140.0)
    max_iterations: int = 5  # more passes for lower AI score
    min_severity: Severity = Severity.LOW
    enable_ai_rewrite: bool = False  # requires API key
    enable_rag: bool = False  # requires corpus
    voice_profile: Optional[str] = None  # personality to inject

    # Statistical targets (from human writing research - Rosenfeld 2024)
    target_sentence_length_std: float = 7.5
    target_vocabulary_richness: float = 0.72
    target_connector_density_max: float = 0.12  # tighter than before
    max_word_repetition: int = 3
