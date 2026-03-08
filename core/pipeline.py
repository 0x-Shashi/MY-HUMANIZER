"""
Main pipeline orchestrator — the 6-pass humanization engine.

DETECT → RETRIEVE → REWRITE → INJECT → VERIFY → ITERATE

This is the brain of Shiro Humanizer. It chains all modules together
and implements the adversarial iteration loop that makes the system
fundamentally different from every existing humanizer.

Key difference from existing tools:
1. Multi-pass (not single-pass) — transforms are verified and iterated
2. Statistically-grounded — targets actual metrics that detectors use
3. Structurally-aware — doesn't just swap words, restructures sentences
4. Voice-injecting — adds human personality, not just neutral rephrasing
5. RAG-grounded — uses real human writing samples as style templates
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from core.types import (
    DetectionResult,
    HumanizationConfig,
    Severity,
    TransformationChange,
    TransformationResult,
    TransformPass,
    WritingDomain,
)
from detection.detector import detect
from retrieval.retriever import HumanCorpusManager
from transformation.burstiness_engine import inject_perplexity, shape_burstiness
from transformation.sentence_level import transform_sentences
from transformation.voice_injector import inject_voice
from transformation.word_level import transform_words
from verification.verifier import (
    VerificationResult,
    get_iteration_guidance,
    verify,
)


@dataclass
class PipelineStats:
    """Telemetry for a pipeline run."""
    total_time_ms: float = 0.0
    detect_time_ms: float = 0.0
    retrieve_time_ms: float = 0.0
    rewrite_time_ms: float = 0.0
    inject_time_ms: float = 0.0
    verify_time_ms: float = 0.0
    iterations: int = 0
    total_changes: int = 0
    ai_score_before: float = 0.0
    ai_score_after: float = 0.0


@dataclass
class PipelineState:
    """Mutable state during pipeline execution."""
    text: str
    changes: list[TransformationChange] = field(default_factory=list)
    detection_before: DetectionResult = field(default_factory=DetectionResult)
    detection_current: DetectionResult = field(default_factory=DetectionResult)
    verification: VerificationResult | None = None
    iteration: int = 0
    rng: random.Random = field(default_factory=random.Random)


class HumanizationPipeline:
    """
    The 6-pass humanization engine.

    Usage:
        pipeline = HumanizationPipeline()
        result = pipeline.run("AI-generated text here...", config)
    """

    def __init__(
        self,
        corpus_manager: HumanCorpusManager | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
    ):
        """
        Args:
            corpus_manager: Optional RAG corpus manager (for Pass 2)
            on_progress: Callback(message, iteration, max_iterations)
        """
        self.corpus = corpus_manager
        self.on_progress = on_progress

    def run(
        self,
        text: str,
        config: HumanizationConfig | None = None,
        seed: int | None = None,
    ) -> TransformationResult:
        """
        Run the full humanization pipeline.

        Args:
            text: Input text (presumably AI-generated)
            config: Humanization settings (defaults used if None)
            seed: Random seed for reproducibility

        Returns:
            TransformationResult with original, transformed text, and diagnostics
        """
        if config is None:
            config = HumanizationConfig()

        start_time = time.perf_counter()
        stats = PipelineStats()

        # Initialize state
        state = PipelineState(
            text=text.strip(),
            rng=random.Random(seed) if seed is not None else random.Random(),
        )

        # ═══════════════════════════════════════════════════════════
        # PASS 1: DETECT — analyze the input
        # ═══════════════════════════════════════════════════════════
        self._report("Analyzing input...", 0, config.max_iterations)
        t0 = time.perf_counter()
        state.detection_before = detect(text, config.min_severity)
        state.detection_current = state.detection_before
        stats.detect_time_ms = (time.perf_counter() - t0) * 1000
        stats.ai_score_before = state.detection_before.ai_score

        # If text doesn't look like AI, return early
        if state.detection_before.ai_score < 0.15:
            return TransformationResult(
                original_text=text,
                transformed_text=text,
                detection_before=state.detection_before,
                detection_after=state.detection_before,
                iterations=0,
                domain=config.domain,
            )

        # ═══════════════════════════════════════════════════════════
        # PASS 2: RETRIEVE — get human writing samples for guidance
        # ═══════════════════════════════════════════════════════════
        t0 = time.perf_counter()
        retrieved_samples = []
        if config.enable_rag and self.corpus:
            self._report("Retrieving human writing samples...", 0, config.max_iterations)
            retrieved_samples = self.corpus.retrieve_similar(text, top_k=3)
        stats.retrieve_time_ms = (time.perf_counter() - t0) * 1000

        # ═══════════════════════════════════════════════════════════
        # PASS 3-4: REWRITE + INJECT (iterative)
        # ═══════════════════════════════════════════════════════════
        guidance = {
            "word_level": True,
            "sentence_level": True,
            "burstiness": True,
            "voice": True,
        }

        for iteration in range(config.max_iterations):
            state.iteration = iteration + 1
            self._report(
                f"Transform pass {state.iteration}...",
                state.iteration,
                config.max_iterations,
            )

            t0 = time.perf_counter()

            # ── PASS 3: REWRITE ─────────────────────────────────
            if guidance.get("word_level", True):
                state.text, word_changes = transform_words(
                    state.text,
                    state.detection_current.patterns,
                    config,
                    state.rng,
                )
                state.changes.extend(word_changes)

            if guidance.get("sentence_level", True):
                state.text, sent_changes = transform_sentences(
                    state.text, config, state.rng,
                )
                state.changes.extend(sent_changes)

            # ── PASS 4: INJECT ──────────────────────────────────
            if guidance.get("burstiness", True):
                state.text, burst_changes = shape_burstiness(
                    state.text, config, state.rng,
                )
                state.changes.extend(burst_changes)

                state.text, perp_changes = inject_perplexity(
                    state.text, config, state.rng,
                )
                state.changes.extend(perp_changes)

            # Voice injection only on first pass (to avoid stacking)
            if iteration == 0 and guidance.get("voice", True):
                if config.voice_profile != "none":
                    state.text = inject_voice(state.text, config, state.rng)

            stats.rewrite_time_ms += (time.perf_counter() - t0) * 1000

            # ═══════════════════════════════════════════════════════
            # PASS 5: VERIFY — check if transformation worked
            # ═══════════════════════════════════════════════════════
            self._report(
                f"Verifying pass {state.iteration}...",
                state.iteration,
                config.max_iterations,
            )
            t0 = time.perf_counter()
            state.verification = verify(
                text,
                state.text,
                config,
                state.detection_current,
            )
            state.detection_current = state.verification.detection
            stats.verify_time_ms += (time.perf_counter() - t0) * 1000

            # ═══════════════════════════════════════════════════════
            # PASS 6: ITERATE — decide if we need another pass
            # ═══════════════════════════════════════════════════════
            if state.verification.passed:
                self._report(
                    f"Passed verification at iteration {state.iteration}!",
                    state.iteration,
                    config.max_iterations,
                )
                break

            # Get guidance on what to fix next
            guidance = get_iteration_guidance(state.verification)

            # If nothing specific to fix, we're done
            if not any(guidance.values()):
                break

        stats.iterations = state.iteration
        stats.total_changes = len(state.changes)
        stats.ai_score_after = state.detection_current.ai_score
        stats.total_time_ms = (time.perf_counter() - start_time) * 1000

        return TransformationResult(
            original_text=text,
            transformed_text=state.text,
            changes=state.changes,
            detection_before=state.detection_before,
            detection_after=state.detection_current,
            iterations=state.iteration,
            domain=config.domain,
        )

    def _report(self, message: str, iteration: int, max_iterations: int) -> None:
        """Send progress update if callback is set."""
        if self.on_progress:
            self.on_progress(message, iteration, max_iterations)
