"""
Lexical AI patterns — word/phrase-level tells.

Sources:
- Kobak et al. (Science Advances 2025): 15M PubMed abstracts, r-ratio measurement
- Liang et al. (Nature Human Behaviour 2024): 950K+ papers
- Neri et al. (2024): Cross-validated excess words
- Juzek & Ward (COLING 2025): RLHF-induced patterns
- Rosenfeld & Lazebnik (2024): Connector/filler detection
- Wikipedia "Signs of AI writing" (WikiProject AI Cleanup)
"""

from __future__ import annotations

from core.types import PatternCategory, PatternMatch, Severity

# ─────────────────────────────────────────────────────────────────────
# Pattern registry: (pattern, replacement, severity, message, source)
# ─────────────────────────────────────────────────────────────────────

LEXICAL_PATTERNS: list[tuple[str, str | None, Severity, str, str]] = [
    # ── CRITICAL (r > 10× baseline from Kobak 2025) ──────────────────
    ("delve", "explore", Severity.CRITICAL,
     "Most extreme LLM outlier (25× baseline)", "Kobak 2025"),
    ("delves", "explores", Severity.CRITICAL,
     "Most extreme LLM outlier (25× baseline)", "Kobak 2025"),
    ("delving", "exploring", Severity.CRITICAL,
     "Most extreme LLM outlier", "Kobak 2025"),

    # Sycophantic openers (RLHF-induced — Juzek 2025)
    ("Certainly!", None, Severity.CRITICAL,
     "Sycophantic opener from RLHF training", "Juzek 2025"),
    ("Of course!", None, Severity.CRITICAL,
     "Sycophantic opener", "Juzek 2025"),
    ("Absolutely!", None, Severity.CRITICAL,
     "Sycophantic opener", "Juzek 2025"),
    ("Great question!", None, Severity.CRITICAL,
     "Sycophantic opener", "Juzek 2025"),
    ("Happy to help", None, Severity.CRITICAL,
     "Chatbot artifact", "Juzek 2025"),
    ("Happy to explain", None, Severity.CRITICAL,
     "Chatbot artifact", "Juzek 2025"),
    ("I'd be happy to", None, Severity.CRITICAL,
     "Chatbot collaboration artifact", "Juzek 2025"),
    ("I would be happy to", None, Severity.CRITICAL,
     "Chatbot collaboration artifact", "Juzek 2025"),

    # Chatbot closers
    ("I hope this helps", None, Severity.CRITICAL,
     "Chatbot closer", "Juzek 2025"),
    ("Let me know if", None, Severity.CRITICAL,
     "Chatbot closer", "Juzek 2025"),
    ("Feel free to", None, Severity.CRITICAL,
     "Chatbot closer", "Juzek 2025"),

    # ── HIGH (r > 3× or clear LLM boilerplate) ──────────────────────
    # Excess adjectives (Kobak 2025 + Liang/Neri 2024)
    ("meticulous", "careful", Severity.HIGH,
     "LLM-excess adjective", "Neri 2024"),
    ("meticulously", "carefully", Severity.HIGH,
     "LLM-excess adverb", "Neri 2024"),
    ("intricate", "complex", Severity.HIGH,
     "Doubled post-2023", "Liang 2024"),
    ("intricacies", "complexities", Severity.HIGH,
     "Doubled post-2023", "Liang 2024"),
    ("pivotal", "key", Severity.HIGH,
     "Doubled post-2023", "Liang 2024"),
    ("showcasing", "showing", Severity.HIGH,
     "LLM-excess 9× frequency", "Kobak 2025"),
    ("underscore", "highlight", Severity.HIGH,
     "9× frequency excess", "Kobak 2025"),
    ("underscores", "highlights", Severity.HIGH,
     "9× frequency excess", "Kobak 2025"),
    ("realm", "area", Severity.HIGH,
     "Doubled post-2023", "Liang 2024"),
    ("tapestry", "fabric", Severity.HIGH,
     "LLM-specific phrase", "Neri 2024"),
    ("testament", "evidence", Severity.HIGH,
     "LLM-specific phrase", "Neri 2024"),
    ("stands as a testament", "serves as evidence", Severity.HIGH,
     "Multi-word LLM tell", "Neri 2024"),

    # Verb inflation (Kobak 2025 high-frequency excess)
    ("leveraging", "using", Severity.HIGH,
     "Verb inflation", "Kobak 2025"),
    ("leverage", "use", Severity.HIGH,
     "Verb inflation", "Kobak 2025"),
    ("utilize", "use", Severity.HIGH,
     "Verb inflation", "Kobak 2025"),
    ("utilizing", "using", Severity.HIGH,
     "Verb inflation", "Kobak 2025"),
    ("utilization", "use", Severity.HIGH,
     "Verb inflation", "Kobak 2025"),
    ("facilitate", "help", Severity.HIGH,
     "Verb inflation", "Kobak 2025"),
    ("facilitating", "helping", Severity.HIGH,
     "Verb inflation", "Kobak 2025"),
    ("commence", "start", Severity.HIGH,
     "Verb inflation", "Kobak 2025"),
    ("endeavor", "try", Severity.HIGH,
     "Verb inflation", "Kobak 2025"),
    ("endeavors", "tries", Severity.HIGH,
     "Verb inflation", "Kobak 2025"),
    ("streamline", "simplify", Severity.HIGH,
     "Verb inflation", "Kobak 2025"),
    ("streamlining", "simplifying", Severity.HIGH,
     "Verb inflation", "Kobak 2025"),
    ("garner", "get", Severity.HIGH,
     "LLM-excess verb", "Wikipedia AI Signs"),
    ("garnered", "got", Severity.HIGH,
     "LLM-excess verb", "Wikipedia AI Signs"),

    # Significance inflation (Wikipedia Signs of AI)
    ("groundbreaking", "important", Severity.HIGH,
     "Promotional language", "Wikipedia AI Signs"),
    ("cutting-edge", "modern", Severity.HIGH,
     "Promotional language", "Wikipedia AI Signs"),
    ("revolutionary", "new", Severity.HIGH,
     "Promotional language", "Wikipedia AI Signs"),
    ("innovative", "new", Severity.HIGH,
     "Promotional language", "Wikipedia AI Signs"),
    ("vibrant", "lively", Severity.HIGH,
     "Promotional/advertisement-like language", "Wikipedia AI Signs"),
    ("nestled", "located", Severity.HIGH,
     "Promotional language", "Wikipedia AI Signs"),
    ("breathtaking", "striking", Severity.HIGH,
     "Promotional language", "Wikipedia AI Signs"),
    ("stunning", "notable", Severity.HIGH,
     "Promotional language", "Wikipedia AI Signs"),
    ("renowned", "well-known", Severity.HIGH,
     "Promotional language", "Wikipedia AI Signs"),

    # Copula avoidance (Wikipedia Signs of AI)
    ("serves as", "is", Severity.HIGH,
     "Copula avoidance — prefer is/are", "Wikipedia AI Signs"),
    ("stands as", "is", Severity.HIGH,
     "Copula avoidance", "Wikipedia AI Signs"),
    ("marks a", "is a", Severity.HIGH,
     "Copula avoidance", "Wikipedia AI Signs"),

    # ── MEDIUM (elevated Δ, pre-LLM marketing share) ────────────────
    ("comprehensive", "thorough", Severity.MEDIUM,
     "Δ=0.041 highest gap", "Kobak 2025"),
    ("crucial", "important", Severity.MEDIUM,
     "LLM-favored adjective", "Kobak 2025"),
    ("robust", "strong", Severity.MEDIUM,
     "Flagged but legitimate in security context", "Kobak 2025"),
    ("seamlessly", "smoothly", Severity.MEDIUM,
     "LLM-excess adverb", "Kobak 2025"),
    ("multifaceted", "complex", Severity.MEDIUM,
     "LLM-excess adjective", "Kobak 2025"),
    ("enhancing", "improving", Severity.MEDIUM,
     "LLM-favored verb", "Kobak 2025"),
    ("harnessing", "using", Severity.MEDIUM,
     "LLM verb inflation", "Kobak 2025"),
    ("evolving landscape", "changing field", Severity.MEDIUM,
     "Marketing cliché", "Kobak 2025"),
    ("landscape", "field", Severity.MEDIUM,
     "Abstract noun overuse in AI text", "Wikipedia AI Signs"),
    ("fostering", "building", Severity.MEDIUM,
     "LLM-favored verb", "Wikipedia AI Signs"),
    ("foster", "build", Severity.MEDIUM,
     "LLM-favored verb", "Wikipedia AI Signs"),
    ("enduring", "lasting", Severity.MEDIUM,
     "LLM-excess adjective", "Wikipedia AI Signs"),
    ("align with", "match", Severity.MEDIUM,
     "LLM-favored phrase", "Wikipedia AI Signs"),
    ("enhance", "improve", Severity.MEDIUM,
     "LLM-favored verb", "Wikipedia AI Signs"),
    ("interplay", "interaction", Severity.MEDIUM,
     "LLM-favored noun", "Wikipedia AI Signs"),
    ("embark", "start", Severity.MEDIUM,
     "LLM verb inflation", "Kobak 2025"),
    ("embarking", "starting", Severity.MEDIUM,
     "LLM verb inflation", "Kobak 2025"),
    ("notably", "especially", Severity.MEDIUM,
     "LLM connector", "Kobak 2025"),

    # ── MEDIUM — Academic report AI tells ───────────────────────────
    ("paramount", "critical", Severity.MEDIUM,
     "LLM-favored adjective in academic text", "Liang 2024"),
    ("unprecedented", "rare", Severity.MEDIUM,
     "LLM-favored promotional adjective", "Liang 2024"),
    ("fundamentally", "essentially", Severity.MEDIUM,
     "LLM-favored adverb", "Kobak 2025"),
    ("paradigm shift", "major shift", Severity.MEDIUM,
     "LLM cliché", "Liang 2024"),
    ("paradigm", "framework", Severity.MEDIUM,
     "LLM-favored abstract noun", "Liang 2024"),
    ("implications", "effects", Severity.MEDIUM,
     "LLM-favored noun", "Liang 2024"),
    ("intersection", "overlap", Severity.MEDIUM,
     "LLM abstract noun", "Liang 2024"),
    ("inherent", "built-in", Severity.MEDIUM,
     "LLM-favored adjective", "Kobak 2025"),
    ("navigated", "handled", Severity.MEDIUM,
     "LLM verb inflation", "Kobak 2025"),
    ("immense", "significant", Severity.MEDIUM,
     "LLM-favored adjective", "Liang 2024"),
    ("delicate balance", "careful balance", Severity.MEDIUM,
     "LLM cliché", "Liang 2024"),
    ("underscores", "highlights", Severity.MEDIUM,
     "LLM-favored verb (25x overuse)", "Kobak 2025"),
    ("underscore", "highlight", Severity.MEDIUM,
     "LLM-favored verb", "Kobak 2025"),
    ("nuanced", "detailed", Severity.MEDIUM,
     "LLM-favored adjective", "Liang 2024"),
    ("noteworthy", "notable", Severity.MEDIUM,
     "LLM-favored adjective", "Liang 2024"),
    ("imperative", "necessary", Severity.MEDIUM,
     "LLM-favored adjective", "Liang 2024"),
    ("bolster", "strengthen", Severity.MEDIUM,
     "LLM-favored verb", "Kobak 2025"),
    ("bolstering", "strengthening", Severity.MEDIUM,
     "LLM-favored verb", "Kobak 2025"),
    ("myriad", "many", Severity.MEDIUM,
     "LLM-favored quantifier", "Liang 2024"),
    ("plethora", "many", Severity.MEDIUM,
     "LLM-favored quantifier", "Liang 2024"),
    ("particularly", "especially", Severity.MEDIUM,
     "Cross-validated excess", "Kobak 2025"),
    ("exhibited", "showed", Severity.MEDIUM,
     "LLM-excess verb", "Kobak 2025"),
    ("boast", "have", Severity.MEDIUM,
     "Copula avoidance", "Wikipedia AI Signs"),
    ("boasts", "has", Severity.MEDIUM,
     "Copula avoidance", "Wikipedia AI Signs"),
    ("ingrained", "embedded", Severity.MEDIUM,
     "LLM-excess adjective", "Kobak 2025"),
    ("indelible", "lasting", Severity.MEDIUM,
     "LLM-excess adjective", "Kobak 2025"),

    # ── LOW (filler, hedging, connectors — Rosenfeld 2024) ──────────
    ("moreover", None, Severity.LOW,
     "Discourse connector overuse", "Rosenfeld 2024"),
    ("furthermore", None, Severity.LOW,
     "Discourse connector overuse", "Rosenfeld 2024"),
    ("additionally", None, Severity.LOW,
     "Discourse connector overuse", "Rosenfeld 2024"),
    ("subsequently", "then", Severity.LOW,
     "Formal connector", "Rosenfeld 2024"),
    ("in conclusion", "overall", Severity.LOW,
     "Formulaic closer", "Rosenfeld 2024"),
    ("consequently", "so", Severity.LOW,
     "Formal connector", "Rosenfeld 2024"),
    ("nevertheless", "still", Severity.LOW,
     "Formal connector", "Rosenfeld 2024"),
    ("nonetheless", "still", Severity.LOW,
     "Formal connector", "Rosenfeld 2024"),

    # Hedging stacks
    ("could potentially", "could", Severity.LOW,
     "Hedging stack", "Rosenfeld 2024"),
    ("might possibly", "might", Severity.LOW,
     "Hedging stack", "Rosenfeld 2024"),

    # Filler phrases
    ("in order to", "to", Severity.LOW,
     "Filler phrase", "Rosenfeld 2024"),
    ("due to the fact that", "because", Severity.LOW,
     "Filler phrase", "Rosenfeld 2024"),
    ("it is worth noting", None, Severity.LOW,
     "Filler phrase", "Rosenfeld 2024"),
    ("it is important to note", None, Severity.LOW,
     "Filler phrase", "Rosenfeld 2024"),
    ("it should be noted that", None, Severity.LOW,
     "Filler phrase", "Rosenfeld 2024"),
    ("serves as a reminder", None, Severity.LOW,
     "Filler phrase", "Rosenfeld 2024"),
    ("at this point in time", "now", Severity.LOW,
     "Filler phrase", "Rosenfeld 2024"),
    ("has the ability to", "can", Severity.LOW,
     "Filler phrase", "Rosenfeld 2024"),
    ("in the realm of", "in", Severity.LOW,
     "Filler phrase combining two LLM tells", "Kobak 2025"),
    ("a myriad of", "many", Severity.LOW,
     "Filler phrase", "Rosenfeld 2024"),
    ("a plethora of", "many", Severity.LOW,
     "Filler phrase", "Rosenfeld 2024"),
]


def detect_lexical_patterns(text: str, min_severity: Severity = Severity.LOW) -> list[PatternMatch]:
    """Scan text for known AI lexical patterns using word-boundary matching."""
    severity_rank = {
        Severity.CRITICAL: 0, Severity.HIGH: 1,
        Severity.MEDIUM: 2, Severity.LOW: 3,
    }
    matches: list[PatternMatch] = []
    text_lower = text.lower()

    for pattern, replacement, severity, message, source in LEXICAL_PATTERNS:
        if severity_rank[severity] > severity_rank[min_severity]:
            continue

        needle = pattern.lower()
        search_start = 0
        while True:
            idx = text_lower.find(needle, search_start)
            if idx == -1:
                break

            # Word boundary checks
            if idx > 0 and text_lower[idx - 1].isalnum():
                search_start = idx + 1
                continue
            end_idx = idx + len(needle)
            if end_idx < len(text_lower) and text_lower[end_idx].isalnum():
                search_start = idx + 1
                continue

            # Skip if inside backtick code spans
            before = text[:idx]
            backtick_count = before.count('`')
            if backtick_count % 2 == 1:
                search_start = idx + 1
                continue

            matched_original = text[idx:end_idx]

            # Apply case-preserving replacement
            final_replacement = None
            if replacement is not None:
                final_replacement = _apply_case(matched_original, replacement)

            matches.append(PatternMatch(
                pattern_id=f"lex_{pattern.lower().replace(' ', '_')}",
                category=PatternCategory.LEXICAL,
                severity=severity,
                start=idx,
                end=end_idx,
                matched_text=matched_original,
                replacement=final_replacement,
                message=message,
                source=source,
            ))
            search_start = end_idx

    # Sort by position, then severity (critical first)
    matches.sort(key=lambda m: (m.start, severity_rank[m.severity]))
    return matches


def _apply_case(original: str, replacement: str) -> str:
    """Preserve the casing of the original text on the replacement."""
    if original.isupper():
        return replacement[0].upper() + replacement[1:]
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement
