"""
Word-level transformation — context-aware replacement of AI-flagged words.

Unlike humanize-ai's static replacement maps, this uses:
1. Multiple replacement candidates per pattern (randomized)
2. Context-aware selection (what fits the surrounding text)
3. Probability-based application (not every match gets replaced)
4. Domain-aware vocabulary (academic vs casual vs technical)

Fixes loophole: "bad always → challenging is a fingerprint itself"
"""

from __future__ import annotations

import random
import re
from typing import Optional

from core.types import (
    HumanizationConfig,
    PatternMatch,
    TransformationChange,
    TransformPass,
    WritingDomain,
)

# ─────────────────────────────────────────────────────────────────────
# Multi-candidate replacement maps (randomized, domain-aware)
# Key: pattern word (lowercase)
# Value: dict[domain, list[replacement_options]]
# ─────────────────────────────────────────────────────────────────────

REPLACEMENTS: dict[str, dict[WritingDomain, list[str]]] = {
    # CRITICAL
    "delve": {
        WritingDomain.ACADEMIC: ["examine", "investigate", "study", "analyze"],
        WritingDomain.CASUAL: ["look into", "dig into", "explore", "check out"],
        WritingDomain.TECHNICAL: ["analyze", "inspect", "examine", "probe"],
        WritingDomain.CREATIVE: ["explore", "wander through", "uncover", "discover"],
        WritingDomain.BUSINESS: ["review", "assess", "evaluate", "examine"],
    },
    "delves": {
        WritingDomain.ACADEMIC: ["examines", "investigates", "studies", "analyzes"],
        WritingDomain.CASUAL: ["looks into", "digs into", "explores"],
        WritingDomain.TECHNICAL: ["analyzes", "inspects", "examines"],
        WritingDomain.CREATIVE: ["explores", "uncovers", "discovers"],
        WritingDomain.BUSINESS: ["reviews", "assesses", "evaluates"],
    },
    "delving": {
        WritingDomain.ACADEMIC: ["examining", "investigating", "studying"],
        WritingDomain.CASUAL: ["looking into", "digging into", "exploring"],
        WritingDomain.TECHNICAL: ["analyzing", "inspecting", "examining"],
        WritingDomain.CREATIVE: ["exploring", "uncovering", "discovering"],
        WritingDomain.BUSINESS: ["reviewing", "assessing", "evaluating"],
    },

    # HIGH — Excess adjectives
    "meticulous": {
        WritingDomain.ACADEMIC: ["careful", "precise", "rigorous", "thorough"],
        WritingDomain.CASUAL: ["careful", "detailed", "thorough"],
        WritingDomain.TECHNICAL: ["precise", "exact", "rigorous"],
        WritingDomain.CREATIVE: ["careful", "attentive", "painstaking"],
        WritingDomain.BUSINESS: ["careful", "detailed", "diligent"],
    },
    "meticulously": {
        WritingDomain.ACADEMIC: ["carefully", "precisely", "rigorously"],
        WritingDomain.CASUAL: ["carefully", "closely", "thoroughly"],
        WritingDomain.TECHNICAL: ["precisely", "exactly", "rigorously"],
        WritingDomain.CREATIVE: ["carefully", "attentively"],
        WritingDomain.BUSINESS: ["carefully", "thoroughly", "diligently"],
    },
    "intricate": {
        WritingDomain.ACADEMIC: ["complex", "detailed", "involved"],
        WritingDomain.CASUAL: ["complicated", "tricky", "tangled"],
        WritingDomain.TECHNICAL: ["complex", "detailed", "elaborate"],
        WritingDomain.CREATIVE: ["layered", "winding", "elaborate"],
        WritingDomain.BUSINESS: ["complex", "detailed", "involved"],
    },
    "pivotal": {
        WritingDomain.ACADEMIC: ["central", "critical", "significant"],
        WritingDomain.CASUAL: ["key", "major", "big"],
        WritingDomain.TECHNICAL: ["critical", "essential", "core"],
        WritingDomain.CREATIVE: ["defining", "central", "turning-point"],
        WritingDomain.BUSINESS: ["key", "critical", "decisive"],
    },
    "showcasing": {
        WritingDomain.ACADEMIC: ["demonstrating", "presenting", "displaying"],
        WritingDomain.CASUAL: ["showing", "displaying", "presenting"],
        WritingDomain.TECHNICAL: ["demonstrating", "presenting", "illustrating"],
        WritingDomain.CREATIVE: ["revealing", "displaying", "presenting"],
        WritingDomain.BUSINESS: ["presenting", "demonstrating", "highlighting"],
    },
    "tapestry": {
        WritingDomain.ACADEMIC: ["fabric", "collection", "mixture"],
        WritingDomain.CASUAL: ["mix", "blend", "collection"],
        WritingDomain.TECHNICAL: ["system", "network", "framework"],
        WritingDomain.CREATIVE: ["weave", "mosaic", "pattern"],
        WritingDomain.BUSINESS: ["mix", "combination", "collection"],
    },
    "testament": {
        WritingDomain.ACADEMIC: ["evidence", "proof", "indication"],
        WritingDomain.CASUAL: ["sign", "proof", "example"],
        WritingDomain.TECHNICAL: ["evidence", "indicator", "proof"],
        WritingDomain.CREATIVE: ["sign", "mark", "witness"],
        WritingDomain.BUSINESS: ["evidence", "demonstration", "proof"],
    },
    "realm": {
        WritingDomain.ACADEMIC: ["field", "domain", "area"],
        WritingDomain.CASUAL: ["area", "world", "space"],
        WritingDomain.TECHNICAL: ["domain", "area", "field"],
        WritingDomain.CREATIVE: ["world", "sphere", "territory"],
        WritingDomain.BUSINESS: ["area", "sector", "space"],
    },

    # HIGH — Verb inflation
    "leverage": {
        WritingDomain.ACADEMIC: ["use", "employ", "apply"],
        WritingDomain.CASUAL: ["use", "take advantage of"],
        WritingDomain.TECHNICAL: ["use", "apply", "employ"],
        WritingDomain.CREATIVE: ["use", "draw on", "tap into"],
        WritingDomain.BUSINESS: ["use", "apply", "capitalize on"],
    },
    "leveraging": {
        WritingDomain.ACADEMIC: ["using", "employing", "applying"],
        WritingDomain.CASUAL: ["using", "taking advantage of"],
        WritingDomain.TECHNICAL: ["using", "applying", "employing"],
        WritingDomain.CREATIVE: ["using", "drawing on", "tapping into"],
        WritingDomain.BUSINESS: ["using", "applying", "capitalizing on"],
    },
    "utilize": {
        WritingDomain.ACADEMIC: ["use", "employ", "apply"],
        WritingDomain.CASUAL: ["use"],
        WritingDomain.TECHNICAL: ["use", "apply"],
        WritingDomain.CREATIVE: ["use", "put to work"],
        WritingDomain.BUSINESS: ["use", "employ"],
    },
    "utilizing": {
        WritingDomain.ACADEMIC: ["using", "employing", "applying"],
        WritingDomain.CASUAL: ["using"],
        WritingDomain.TECHNICAL: ["using", "applying"],
        WritingDomain.CREATIVE: ["using", "putting to work"],
        WritingDomain.BUSINESS: ["using", "employing"],
    },
    "facilitate": {
        WritingDomain.ACADEMIC: ["enable", "support", "allow"],
        WritingDomain.CASUAL: ["help", "make easier", "allow"],
        WritingDomain.TECHNICAL: ["enable", "support", "allow"],
        WritingDomain.CREATIVE: ["help", "enable", "open the way for"],
        WritingDomain.BUSINESS: ["help", "enable", "support"],
    },
    "commence": {
        WritingDomain.ACADEMIC: ["begin", "start", "initiate"],
        WritingDomain.CASUAL: ["start", "begin", "kick off"],
        WritingDomain.TECHNICAL: ["start", "begin", "initiate"],
        WritingDomain.CREATIVE: ["begin", "start", "set out"],
        WritingDomain.BUSINESS: ["start", "begin", "launch"],
    },
    "endeavor": {
        WritingDomain.ACADEMIC: ["attempt", "effort", "undertaking"],
        WritingDomain.CASUAL: ["try", "effort", "attempt"],
        WritingDomain.TECHNICAL: ["attempt", "effort"],
        WritingDomain.CREATIVE: ["attempt", "venture", "pursuit"],
        WritingDomain.BUSINESS: ["effort", "initiative", "attempt"],
    },
    "streamline": {
        WritingDomain.ACADEMIC: ["simplify", "optimize", "improve"],
        WritingDomain.CASUAL: ["simplify", "clean up", "make easier"],
        WritingDomain.TECHNICAL: ["optimize", "simplify", "improve"],
        WritingDomain.CREATIVE: ["simplify", "smooth out", "trim"],
        WritingDomain.BUSINESS: ["simplify", "optimize", "improve"],
    },
    "harnessing": {
        WritingDomain.ACADEMIC: ["using", "applying", "employing"],
        WritingDomain.CASUAL: ["using", "tapping into"],
        WritingDomain.TECHNICAL: ["using", "applying", "deploying"],
        WritingDomain.CREATIVE: ["using", "channeling", "tapping into"],
        WritingDomain.BUSINESS: ["using", "applying", "deploying"],
    },

    # HIGH — Promotional
    "groundbreaking": {
        WritingDomain.ACADEMIC: ["significant", "novel", "important"],
        WritingDomain.CASUAL: ["big", "major", "impressive"],
        WritingDomain.TECHNICAL: ["novel", "significant", "new"],
        WritingDomain.CREATIVE: ["bold", "fresh", "daring"],
        WritingDomain.BUSINESS: ["significant", "major", "important"],
    },
    "cutting-edge": {
        WritingDomain.ACADEMIC: ["advanced", "modern", "recent"],
        WritingDomain.CASUAL: ["latest", "newest", "modern"],
        WritingDomain.TECHNICAL: ["state-of-the-art", "advanced", "latest"],
        WritingDomain.CREATIVE: ["fresh", "forward-looking", "modern"],
        WritingDomain.BUSINESS: ["advanced", "leading", "modern"],
    },
    "revolutionary": {
        WritingDomain.ACADEMIC: ["transformative", "significant", "novel"],
        WritingDomain.CASUAL: ["game-changing", "huge", "major"],
        WritingDomain.TECHNICAL: ["transformative", "paradigm-shifting", "novel"],
        WritingDomain.CREATIVE: ["radical", "transformative", "bold"],
        WritingDomain.BUSINESS: ["transformative", "disruptive", "significant"],
    },

    # MEDIUM — Connectors & filler
    "comprehensive": {
        WritingDomain.ACADEMIC: ["thorough", "extensive", "detailed"],
        WritingDomain.CASUAL: ["full", "complete", "broad"],
        WritingDomain.TECHNICAL: ["complete", "full", "thorough"],
        WritingDomain.CREATIVE: ["sweeping", "broad", "full"],
        WritingDomain.BUSINESS: ["complete", "full", "detailed"],
    },
    "crucial": {
        WritingDomain.ACADEMIC: ["important", "essential", "vital"],
        WritingDomain.CASUAL: ["important", "key", "big"],
        WritingDomain.TECHNICAL: ["critical", "essential", "important"],
        WritingDomain.CREATIVE: ["vital", "essential", "critical"],
        WritingDomain.BUSINESS: ["important", "essential", "key"],
    },
    "robust": {
        WritingDomain.ACADEMIC: ["strong", "solid", "reliable"],
        WritingDomain.CASUAL: ["strong", "solid", "tough"],
        WritingDomain.TECHNICAL: ["reliable", "resilient", "fault-tolerant"],
        WritingDomain.CREATIVE: ["sturdy", "strong", "solid"],
        WritingDomain.BUSINESS: ["strong", "reliable", "solid"],
    },
    "innovative": {
        WritingDomain.ACADEMIC: ["novel", "original", "new"],
        WritingDomain.CASUAL: ["new", "fresh", "creative"],
        WritingDomain.TECHNICAL: ["novel", "new", "original"],
        WritingDomain.CREATIVE: ["inventive", "original", "fresh"],
        WritingDomain.BUSINESS: ["new", "creative", "original"],
    },
    "enhance": {
        WritingDomain.ACADEMIC: ["improve", "strengthen", "augment"],
        WritingDomain.CASUAL: ["improve", "boost", "help"],
        WritingDomain.TECHNICAL: ["improve", "optimize", "increase"],
        WritingDomain.CREATIVE: ["enrich", "improve", "deepen"],
        WritingDomain.BUSINESS: ["improve", "strengthen", "boost"],
    },
    "enhancing": {
        WritingDomain.ACADEMIC: ["improving", "strengthening"],
        WritingDomain.CASUAL: ["improving", "boosting"],
        WritingDomain.TECHNICAL: ["improving", "optimizing"],
        WritingDomain.CREATIVE: ["enriching", "deepening"],
        WritingDomain.BUSINESS: ["improving", "strengthening"],
    },
    "fostering": {
        WritingDomain.ACADEMIC: ["building", "promoting", "encouraging"],
        WritingDomain.CASUAL: ["building", "growing", "encouraging"],
        WritingDomain.TECHNICAL: ["enabling", "promoting", "supporting"],
        WritingDomain.CREATIVE: ["nurturing", "cultivating", "growing"],
        WritingDomain.BUSINESS: ["building", "developing", "promoting"],
    },
    "seamlessly": {
        WritingDomain.ACADEMIC: ["smoothly", "naturally", "easily"],
        WritingDomain.CASUAL: ["smoothly", "easily", "without a hitch"],
        WritingDomain.TECHNICAL: ["smoothly", "transparently", "without interruption"],
        WritingDomain.CREATIVE: ["fluidly", "effortlessly", "naturally"],
        WritingDomain.BUSINESS: ["smoothly", "efficiently", "easily"],
    },

    # ── ACADEMIC REPORT SPECIFIC ────────────────────────────────────
    # Words AI constantly uses in college-level writing
    "paramount": {
        WritingDomain.ACADEMIC: ["critical", "essential", "central"],
        WritingDomain.CASUAL: ["really important", "key", "major"],
        WritingDomain.TECHNICAL: ["essential", "critical", "key"],
        WritingDomain.CREATIVE: ["vital", "central", "key"],
        WritingDomain.BUSINESS: ["essential", "important", "key"],
    },
    "multifaceted": {
        WritingDomain.ACADEMIC: ["complex", "varied", "diverse"],
        WritingDomain.CASUAL: ["complicated", "mixed", "varied"],
        WritingDomain.TECHNICAL: ["complex", "multi-dimensional", "varied"],
        WritingDomain.CREATIVE: ["layered", "many-sided", "complex"],
        WritingDomain.BUSINESS: ["complex", "varied", "diverse"],
    },
    "paradigm": {
        WritingDomain.ACADEMIC: ["framework", "model", "approach"],
        WritingDomain.CASUAL: ["way of thinking", "model", "approach"],
        WritingDomain.TECHNICAL: ["model", "framework", "methodology"],
        WritingDomain.CREATIVE: ["worldview", "lens", "framework"],
        WritingDomain.BUSINESS: ["model", "approach", "framework"],
    },
    "paradigm shift": {
        WritingDomain.ACADEMIC: ["fundamental change", "major shift", "transformation"],
        WritingDomain.CASUAL: ["big change", "shake-up", "shift"],
        WritingDomain.TECHNICAL: ["architectural change", "fundamental shift"],
        WritingDomain.CREATIVE: ["sea change", "transformation", "revolution"],
        WritingDomain.BUSINESS: ["major shift", "transformation", "strategic change"],
    },
    "unprecedented": {
        WritingDomain.ACADEMIC: ["rare", "unusual", "remarkable", "new"],
        WritingDomain.CASUAL: ["never-before-seen", "brand new", "huge"],
        WritingDomain.TECHNICAL: ["novel", "first-of-its-kind", "new"],
        WritingDomain.CREATIVE: ["unheard-of", "extraordinary", "brand new"],
        WritingDomain.BUSINESS: ["exceptional", "remarkable", "new"],
    },
    "landscape": {
        WritingDomain.ACADEMIC: ["field", "environment", "context", "area"],
        WritingDomain.CASUAL: ["scene", "picture", "situation"],
        WritingDomain.TECHNICAL: ["environment", "ecosystem", "space"],
        WritingDomain.CREATIVE: ["terrain", "world", "scene"],
        WritingDomain.BUSINESS: ["market", "environment", "sector"],
    },
    "implications": {
        WritingDomain.ACADEMIC: ["effects", "consequences", "results"],
        WritingDomain.CASUAL: ["effects", "results", "impact"],
        WritingDomain.TECHNICAL: ["effects", "consequences", "impacts"],
        WritingDomain.CREATIVE: ["consequences", "ripple effects", "results"],
        WritingDomain.BUSINESS: ["effects", "impacts", "consequences"],
    },
    "intersection": {
        WritingDomain.ACADEMIC: ["overlap", "connection", "convergence"],
        WritingDomain.CASUAL: ["overlap", "meeting point", "crossover"],
        WritingDomain.TECHNICAL: ["overlap", "interface", "convergence"],
        WritingDomain.CREATIVE: ["crossroads", "meeting point", "junction"],
        WritingDomain.BUSINESS: ["overlap", "convergence", "connection"],
    },
    "inherent": {
        WritingDomain.ACADEMIC: ["built-in", "natural", "underlying"],
        WritingDomain.CASUAL: ["natural", "basic", "built-in"],
        WritingDomain.TECHNICAL: ["built-in", "native", "intrinsic"],
        WritingDomain.CREATIVE: ["deep-rooted", "native", "woven-in"],
        WritingDomain.BUSINESS: ["natural", "built-in", "fundamental"],
    },
    "navigated": {
        WritingDomain.ACADEMIC: ["handled", "managed", "addressed"],
        WritingDomain.CASUAL: ["dealt with", "handled", "worked through"],
        WritingDomain.TECHNICAL: ["managed", "handled", "resolved"],
        WritingDomain.CREATIVE: ["traversed", "worked through", "steered through"],
        WritingDomain.BUSINESS: ["managed", "addressed", "handled"],
    },
    "immense": {
        WritingDomain.ACADEMIC: ["considerable", "significant", "great"],
        WritingDomain.CASUAL: ["huge", "massive", "big"],
        WritingDomain.TECHNICAL: ["significant", "substantial", "large"],
        WritingDomain.CREATIVE: ["vast", "enormous", "towering"],
        WritingDomain.BUSINESS: ["significant", "substantial", "major"],
    },
    "delicate balance": {
        WritingDomain.ACADEMIC: ["careful balance", "tension", "trade-off"],
        WritingDomain.CASUAL: ["tricky balance", "fine line", "balancing act"],
        WritingDomain.TECHNICAL: ["trade-off", "equilibrium", "balance"],
        WritingDomain.CREATIVE: ["tightrope", "tense balance", "fine line"],
        WritingDomain.BUSINESS: ["balance", "trade-off", "equilibrium"],
    },
    "fundamentally": {
        WritingDomain.ACADEMIC: ["at its core", "essentially", "deeply"],
        WritingDomain.CASUAL: ["basically", "at its core", "really"],
        WritingDomain.TECHNICAL: ["at a fundamental level", "essentially", "inherently"],
        WritingDomain.CREATIVE: ["at its heart", "deeply", "at the root"],
        WritingDomain.BUSINESS: ["essentially", "at its core", "primarily"],
    },
    "underscores": {
        WritingDomain.ACADEMIC: ["highlights", "shows", "reveals"],
        WritingDomain.CASUAL: ["shows", "points out", "makes clear"],
        WritingDomain.TECHNICAL: ["highlights", "demonstrates", "indicates"],
        WritingDomain.CREATIVE: ["reveals", "exposes", "brings to light"],
        WritingDomain.BUSINESS: ["highlights", "emphasizes", "shows"],
    },
    "underscore": {
        WritingDomain.ACADEMIC: ["highlight", "emphasize", "show"],
        WritingDomain.CASUAL: ["show", "point out", "stress"],
        WritingDomain.TECHNICAL: ["highlight", "demonstrate", "indicate"],
        WritingDomain.CREATIVE: ["reveal", "expose", "bring to light"],
        WritingDomain.BUSINESS: ["highlight", "emphasize", "demonstrate"],
    },
    "myriad": {
        WritingDomain.ACADEMIC: ["many", "numerous", "various"],
        WritingDomain.CASUAL: ["lots of", "tons of", "many"],
        WritingDomain.TECHNICAL: ["numerous", "many", "multiple"],
        WritingDomain.CREATIVE: ["countless", "endless", "many"],
        WritingDomain.BUSINESS: ["many", "numerous", "various"],
    },
    "plethora": {
        WritingDomain.ACADEMIC: ["abundance", "wide range", "many"],
        WritingDomain.CASUAL: ["lots", "bunch", "ton"],
        WritingDomain.TECHNICAL: ["many", "wide range", "large number"],
        WritingDomain.CREATIVE: ["wealth", "flood", "abundance"],
        WritingDomain.BUSINESS: ["many", "wide range", "variety"],
    },
    "noteworthy": {
        WritingDomain.ACADEMIC: ["notable", "significant", "worth mentioning"],
        WritingDomain.CASUAL: ["worth noting", "interesting", "notable"],
        WritingDomain.TECHNICAL: ["significant", "important", "notable"],
        WritingDomain.CREATIVE: ["striking", "remarkable", "interesting"],
        WritingDomain.BUSINESS: ["significant", "important", "notable"],
    },
    "imperative": {
        WritingDomain.ACADEMIC: ["necessary", "essential", "important"],
        WritingDomain.CASUAL: ["a must", "necessary", "crucial"],
        WritingDomain.TECHNICAL: ["necessary", "required", "essential"],
        WritingDomain.CREATIVE: ["urgent", "pressing", "necessary"],
        WritingDomain.BUSINESS: ["necessary", "essential", "critical"],
    },
    "bolster": {
        WritingDomain.ACADEMIC: ["strengthen", "support", "reinforce"],
        WritingDomain.CASUAL: ["boost", "back up", "support"],
        WritingDomain.TECHNICAL: ["reinforce", "strengthen", "improve"],
        WritingDomain.CREATIVE: ["fortify", "prop up", "shore up"],
        WritingDomain.BUSINESS: ["strengthen", "reinforce", "support"],
    },
    "bolstering": {
        WritingDomain.ACADEMIC: ["strengthening", "supporting", "reinforcing"],
        WritingDomain.CASUAL: ["boosting", "backing up", "supporting"],
        WritingDomain.TECHNICAL: ["reinforcing", "strengthening", "improving"],
        WritingDomain.CREATIVE: ["fortifying", "propping up"],
        WritingDomain.BUSINESS: ["strengthening", "reinforcing", "supporting"],
    },
    "nuanced": {
        WritingDomain.ACADEMIC: ["subtle", "detailed", "refined"],
        WritingDomain.CASUAL: ["complicated", "tricky", "detailed"],
        WritingDomain.TECHNICAL: ["detailed", "fine-grained", "subtle"],
        WritingDomain.CREATIVE: ["layered", "textured", "subtle"],
        WritingDomain.BUSINESS: ["detailed", "sophisticated", "subtle"],
    },
    "embark": {
        WritingDomain.ACADEMIC: ["begin", "start", "undertake"],
        WritingDomain.CASUAL: ["start", "kick off", "dive into"],
        WritingDomain.TECHNICAL: ["begin", "initiate", "start"],
        WritingDomain.CREATIVE: ["set out", "venture", "launch into"],
        WritingDomain.BUSINESS: ["begin", "launch", "start"],
    },
}

# Filler phrase replacements (multi-word → shorter form)
FILLER_REPLACEMENTS: dict[str, str] = {
    "in order to": "to",
    "due to the fact that": "because",
    "at this point in time": "now",
    "has the ability to": "can",
    "in the realm of": "in",
    "a myriad of": "many",
    "a plethora of": "many",
    "could potentially": "could",
    "might possibly": "might",
    # Academic report fillers AI overuses
    "it is crucial to understand that": "",
    "it is crucial to understand the": "",
    "it is crucial to understand": "",
    "it is important to understand that": "",
    "it is important to understand the": "",
    "it is important to note that": "",
    "it is worth noting that": "",
    "it should be noted that": "",
    "it is essential to recognize that": "",
    "it is essential to understand that": "",
    "it is essential to understand the": "",
    "it is essential to understand": "",
    "it is essential to note that": "",
    "it goes without saying that": "",
    "in today's rapidly evolving": "in today's",
    "in the ever-evolving landscape of": "in",
    "plays a crucial role in": "matters for",
    "plays a vital role in": "matters for",
    "plays an important role in": "matters for",
    "serves as a testament to": "shows",
    "shed light on": "clarify",
    "sheds light on": "clarifies",
    "pave the way for": "lead to",
    "paves the way for": "leads to",
    "a wide range of": "many",
    "a broad range of": "many",
    "the fact that": "",
    "on the other hand": "but",
    "at the end of the day": "ultimately",
}

# Copula avoidance fixes
COPULA_FIXES: dict[str, str] = {
    "serves as": "is",
    "stands as": "is",
    "represents a": "is a",
    "marks a": "is a",
}


def transform_words(
    text: str,
    patterns: list[PatternMatch],
    config: HumanizationConfig,
    rng: Optional[random.Random] = None,
) -> tuple[str, list[TransformationChange]]:
    """
    Apply word-level transformations based on detected patterns.

    Unlike static maps, this:
    - Picks from multiple candidates randomly
    - Skips some replacements probabilistically (more human)
    - Uses domain-aware vocabulary
    - Applies filler phrase compression
    - Fixes copula avoidance
    """
    if rng is None:
        rng = random.Random()

    changes: list[TransformationChange] = []

    # Process patterns in reverse order (high offset first) to preserve positions
    sorted_patterns = sorted(patterns, key=lambda p: p.start, reverse=True)

    # Track which regions we've already modified to avoid overlapping changes
    modified_regions: list[tuple[int, int]] = []

    for pattern in sorted_patterns:
        # Skip if overlapping with already-modified region
        if any(
            pattern.start < end and pattern.end > start
            for start, end in modified_regions
        ):
            continue

        # Probabilistic skip: don't replace everything (more human)
        # Higher creativity = more replacements
        # For academic domain, replace more aggressively
        base_prob = 0.7 if config.domain == WritingDomain.ACADEMIC else 0.6
        replace_prob = base_prob + 0.28 * config.creativity
        if rng.random() > replace_prob:
            continue

        replacement = _get_replacement(pattern, config.domain, rng)
        if replacement is None:
            continue

        # Apply the replacement
        text = text[:pattern.start] + replacement + text[pattern.end:]

        changes.append(TransformationChange(
            original=pattern.matched_text,
            replacement=replacement,
            start=pattern.start,
            end=pattern.end,
            pass_name=TransformPass.REWRITE,
            reason=f"AI pattern: {pattern.message}",
            confidence=0.9,
        ))
        modified_regions.append((pattern.start, pattern.start + len(replacement)))

    # Apply filler phrase compression (not pattern-based, direct text scan)
    text, filler_changes = _compress_fillers(text, rng, config.creativity)
    changes.extend(filler_changes)

    # Fix copula avoidance
    text, copula_changes = _fix_copula(text, rng, config.creativity)
    changes.extend(copula_changes)

    # Post-cleanup: fix capitalization after sentence-start connector + comma
    # e.g., ". Still, These tools" → ". Still, these tools"
    # Only targets common words (not proper nouns) after connector-comma patterns
    connector_comma_words = {
        "still", "however", "yet", "though", "so", "but", "and",
        "also", "thus", "hence", "overall", "arguably", "perhaps",
        "interestingly", "admittedly", "surprisingly",
    }
    for conn in connector_comma_words:
        pattern = re.compile(
            rf'([.!?]\s+)({re.escape(conn)}),\s+([A-Z])([a-z])',
            re.IGNORECASE,
        )
        text = pattern.sub(
            lambda m: f"{m.group(1)}{m.group(2)}, {m.group(3).lower()}{m.group(4)}",
            text,
        )

    return text, changes


def _get_replacement(
    pattern: PatternMatch,
    domain: WritingDomain,
    rng: random.Random,
) -> Optional[str]:
    """Get a domain-aware replacement for a detected pattern."""
    key = pattern.matched_text.lower().strip()

    # Check multi-candidate replacements first
    if key in REPLACEMENTS:
        domain_options = REPLACEMENTS[key]
        options = domain_options.get(domain, domain_options.get(WritingDomain.CASUAL, []))
        if options:
            chosen = rng.choice(options)
            return _apply_case(pattern.matched_text, chosen)

    # Fall back to pattern's own replacement
    if pattern.replacement:
        return pattern.replacement

    return None


def _compress_fillers(
    text: str, rng: random.Random, creativity: float
) -> tuple[str, list[TransformationChange]]:
    """Replace filler phrases with shorter equivalents."""
    changes: list[TransformationChange] = []

    for filler, replacement in FILLER_REPLACEMENTS.items():
        pattern = re.compile(re.escape(filler), re.IGNORECASE)
        for match in reversed(list(pattern.finditer(text))):
            # Probabilistic: compress more with higher creativity
            if rng.random() > 0.5 + 0.4 * creativity:
                continue

            original = match.group()
            final = _apply_case(original, replacement)

            # When replacement is empty and it's at sentence start,
            # capitalize the next word — but check the result makes sense
            after = text[match.end():]
            stripped_after = after.lstrip(" ,;")
            if not final and stripped_after:
                # Check if the remaining text starts a valid sentence
                first_remaining = stripped_after.split()[0].lower() if stripped_after.split() else ""
                sentence_starters = {
                    "the", "a", "an", "this", "that", "these", "those",
                    "it", "they", "we", "he", "she", "there", "here",
                    "while", "although", "when", "if", "as", "some",
                    "many", "most", "all", "no", "not", "each", "every",
                    "such", "one", "both", "several", "few",
                }
                if first_remaining not in sentence_starters:
                    # Would create an orphan fragment — skip
                    continue
                capitalized_after = stripped_after[0].upper() + stripped_after[1:]
                text = text[:match.start()] + capitalized_after
            else:
                text = text[:match.start()] + final + text[match.end():]

            changes.append(TransformationChange(
                original=original,
                replacement=final if final else "(removed)",
                start=match.start(),
                end=match.end(),
                pass_name=TransformPass.REWRITE,
                reason="Filler phrase compression",
                confidence=0.95,
            ))

    return text, changes


def _fix_copula(
    text: str, rng: random.Random, creativity: float
) -> tuple[str, list[TransformationChange]]:
    """Fix copula avoidance (serves as → is)."""
    changes: list[TransformationChange] = []

    for copula, fix in COPULA_FIXES.items():
        pattern = re.compile(re.escape(copula), re.IGNORECASE)
        for match in reversed(list(pattern.finditer(text))):
            if rng.random() > 0.7 + 0.25 * creativity:
                continue

            original = match.group()
            final = _apply_case(original, fix)
            text = text[:match.start()] + final + text[match.end():]
            changes.append(TransformationChange(
                original=original,
                replacement=final,
                start=match.start(),
                end=match.end(),
                pass_name=TransformPass.REWRITE,
                reason="Copula avoidance fix",
                confidence=0.9,
            ))

    return text, changes


def _apply_case(original: str, replacement: str) -> str:
    """Preserve casing from original onto replacement."""
    if not original or not replacement:
        return replacement
    if original[0].isupper() and not replacement[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement
