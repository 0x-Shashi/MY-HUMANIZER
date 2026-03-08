"""
Voice and personality injector — makes text sound like a PERSON wrote it.

This addresses the fundamental loophole that NO existing humanizer handles:
"Sterile, voiceless writing is just as obviously AI as slop."
(humanizer/SKILL.md insight)

What AI text lacks:
- Opinions, judgments, value statements
- Hedging, uncertainty, self-correction
- Tangential thinking, asides, digressions
- Imperfect structure (fragments, run-ons, one-word paragraphs)
- First-person perspective and experience
- Humor, sarcasm, exaggeration
- Domain-specific jargon and colloquialisms

Approach:
1. Inject hedges and qualifiers ("I think", "probably", "honestly")
2. Add personal markers ("in my experience", "from what I've seen")
3. Insert asides and tangents (parenthetical thoughts)
4. Add opinion markers ("the real issue is", "what bugs me about")
5. Use contractions and informal phrasing
6. Break perfect structure (intentional fragments)
"""

from __future__ import annotations

import random
import re

from core.types import HumanizationConfig, WritingDomain


# ── Hedges and qualifiers by domain ─────────────────────────────────

HEDGES: dict[WritingDomain, list[str]] = {
    WritingDomain.CASUAL: [
        "I think", "probably", "honestly", "I guess", "basically",
        "like", "sort of", "kind of", "I mean", "to be fair",
        "not gonna lie", "tbh", "in a way", "more or less",
    ],
    WritingDomain.ACADEMIC: [
        "arguably", "to some extent", "it appears that",
        "one could argue", "this suggests that",
        "the evidence suggests that", "perhaps",
        "interestingly", "from what the data shows",
        "a closer look reveals that", "in practice",
    ],
    WritingDomain.TECHNICAL: [
        "in practice", "as far as I can tell", "from what I've seen",
        "typically", "in most cases", "it depends on",
        "roughly speaking", "the tradeoff is",
    ],
    WritingDomain.CREATIVE: [
        "something about", "there's a quality to", "you could say",
        "imagine", "the thing is", "call it what you will",
        "in a sense", "strangely enough",
    ],
    WritingDomain.BUSINESS: [
        "in my experience", "from a practical standpoint",
        "the reality is", "what we've found", "generally speaking",
        "all things considered", "the key takeaway is",
    ],
}

# ── Personal experience markers ─────────────────────────────────────

PERSONAL_MARKERS: dict[WritingDomain, list[str]] = {
    WritingDomain.CASUAL: [
        "I've noticed that", "from what I've seen,",
        "in my experience,", "last time I checked,",
        "when I tried this,", "what worked for me was",
    ],
    WritingDomain.ACADEMIC: [
        "in our analysis,", "as the data suggests,",
        "our findings indicate", "upon closer examination,",
    ],
    WritingDomain.TECHNICAL: [
        "when I ran into this,", "the approach I've used is",
        "in the projects I've worked on,", "what helped us was",
        "the gotcha here is", "one thing to watch for:",
    ],
    WritingDomain.CREATIVE: [
        "what struck me was", "I keep coming back to",
        "there's something about", "the way I see it,",
    ],
    WritingDomain.BUSINESS: [
        "what we learned was", "the lesson here is",
        "based on our experience,", "looking at the data,",
    ],
}

# ── Opinion/judgment phrases ────────────────────────────────────────

OPINION_MARKERS: dict[WritingDomain, list[str]] = {
    WritingDomain.CASUAL: [
        "the real problem is", "what bugs me about this is",
        "the cool thing is", "the annoying part is",
        "what people miss is", "here's the thing though —",
    ],
    WritingDomain.ACADEMIC: [
        "a critical gap in the literature is",
        "what is often overlooked is",
        "a more nuanced view suggests",
    ],
    WritingDomain.TECHNICAL: [
        "the elephant in the room is", "what gets overlooked is",
        "the real bottleneck is", "where this falls apart is",
    ],
    WritingDomain.CREATIVE: [
        "the irony is", "what fascinates me is",
        "the beauty of it is", "the cruel truth is",
    ],
    WritingDomain.BUSINESS: [
        "the bottom line is", "what drives results is",
        "the overlooked factor is", "where organizations stumble is",
    ],
}

# ── Parenthetical asides ───────────────────────────────────────────

ASIDES: dict[WritingDomain, list[str]] = {
    WritingDomain.CASUAL: [
        "— and this is important —",
        "— which is often missed —",
        "(at least in my case)",
        "(though your mileage may vary)",
        "(and I'm being generous here)",
        "(surprisingly enough)",
        "— or so I thought —",
        "(bear with me here)",
        ", if that makes sense,",
        "— and I've tested this —",
        "(admittedly)",
        "— no pun intended —",
        "(for better or worse)",
    ],
    WritingDomain.ACADEMIC: [
        "(admittedly)",
        "(in practice)",
        "— a point often overlooked —",
        "(at least in this context)",
        "— though evidence varies —",
        "(to varying degrees)",
        "(with some exceptions)",
        ", broadly speaking,",
    ],
    WritingDomain.TECHNICAL: [
        "(in practice)",
        "(admittedly)",
        "— depending on the setup —",
        "(roughly speaking)",
        "— and this matters —",
        "(with caveats)",
    ],
    WritingDomain.CREATIVE: [
        "— and this is important —",
        "— which is often missed —",
        "(surprisingly enough)",
        "— or so I thought —",
        "(for better or worse)",
        ", oddly enough,",
    ],
    WritingDomain.BUSINESS: [
        "(admittedly)",
        "(in practice)",
        "— and this matters —",
        "(broadly speaking)",
        "(with some exceptions)",
    ],
}

# ── Contractions map ────────────────────────────────────────────────

CONTRACTIONS = {
    "I am": "I'm",
    "you are": "you're",
    "they are": "they're",
    "we are": "we're",
    "it is": "it's",
    "that is": "that's",
    "there is": "there's",
    "what is": "what's",
    "who is": "who's",
    "do not": "don't",
    "does not": "doesn't",
    "did not": "didn't",
    "is not": "isn't",
    "are not": "aren't",
    "was not": "wasn't",
    "were not": "weren't",
    "will not": "won't",
    "would not": "wouldn't",
    "could not": "couldn't",
    "should not": "shouldn't",
    "can not": "can't",
    "cannot": "can't",
    "have not": "haven't",
    "has not": "hasn't",
    "had not": "hadn't",
    "I have": "I've",
    "you have": "you've",
    "they have": "they've",
    "we have": "we've",
    "I will": "I'll",
    "you will": "you'll",
    "they will": "they'll",
    "we will": "we'll",
    "I would": "I'd",
    "you would": "you'd",
    "they would": "they'd",
    "we would": "we'd",
    "let us": "let's",
}


def inject_voice(
    text: str,
    config: HumanizationConfig,
    rng: random.Random | None = None,
) -> str:
    """
    Inject human voice and personality into text.

    This is NOT about replacing AI words — it's about adding the human
    elements that AI fundamentally cannot produce: opinions, tangents,
    doubt, experience, and imperfect structure.

    Args:
        text: Input text (already word/sentence transformed)
        config: Humanization configuration
        rng: Random instance for reproducibility

    Returns:
        Text with human voice injected
    """
    if rng is None:
        rng = random.Random()

    domain = config.domain
    creativity = config.creativity

    # Phase 1: Apply contractions (makes text sound natural)
    text = _apply_contractions(text, domain, rng, creativity)

    # Phase 2: Inject hedges at sentence starts
    text = _inject_hedges(text, domain, rng, creativity)

    # Phase 3: Add personal experience markers
    text = _inject_personal_markers(text, domain, rng, creativity)

    # Phase 4: Insert parenthetical asides
    text = _inject_asides(text, domain, rng, creativity)

    # Phase 5: Add opinion markers at paragraph starts
    text = _inject_opinions(text, domain, rng, creativity)

    return text


def _apply_contractions(
    text: str, domain: WritingDomain, rng: random.Random, creativity: float
) -> str:
    """Apply contractions to make text sound natural."""
    # Academic writing uses fewer contractions
    if domain == WritingDomain.ACADEMIC:
        contraction_rate = 0.2 + creativity * 0.1
    else:
        contraction_rate = 0.5 + creativity * 0.3

    result = text
    for full, contracted in CONTRACTIONS.items():
        if rng.random() < contraction_rate:
            # Case-insensitive replacement with case preservation
            pattern = re.compile(re.escape(full), re.IGNORECASE)
            matches = list(pattern.finditer(result))
            for match in reversed(matches):
                if rng.random() < contraction_rate:
                    original = match.group()
                    # Preserve case of first letter
                    if original[0].isupper():
                        replacement = contracted[0].upper() + contracted[1:]
                    else:
                        replacement = contracted
                    result = result[:match.start()] + replacement + result[match.end():]

    return result


def _inject_hedges(
    text: str, domain: WritingDomain, rng: random.Random, creativity: float
) -> str:
    """Inject hedging phrases at some sentence starts."""
    hedges = HEDGES.get(domain, HEDGES[WritingDomain.CASUAL])
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) < 3:
        return text

    # Inject 1-2 hedges per ~5 sentences
    inject_rate = 0.15 + creativity * 0.15
    result_sents = []
    used_hedges: set[str] = set()  # Avoid repeating the same hedge

    for i, sent in enumerate(sentences):
        if i == 0:
            # Don't hedge the first sentence
            result_sents.append(sent)
            continue

        if rng.random() < inject_rate and len(sent.split()) > 5:
            # Pick a hedge we haven't used yet
            available = [h for h in hedges if h not in used_hedges]
            if not available:
                available = hedges  # Reset if all used
            hedge = rng.choice(available)
            used_hedges.add(hedge)
            # Lowercase the sentence start if we're prepending
            # But skip if sentence starts with a connector (it'll look odd)
            first_word = sent.split()[0].lower().rstrip(",;")
            skip_words = {
                "moreover", "furthermore", "additionally", "however",
                "therefore", "consequently", "nevertheless", "thus",
                "overall", "arguably", "perhaps", "interestingly",
                "admittedly", "surprisingly", "still", "yet",
                "in", "to",  # "in practice", "to some extent"
            }
            if first_word in skip_words:
                result_sents.append(sent)
                continue
            sent_modified = sent[0].lower() + sent[1:] if sent[0].isupper() else sent
            # Determine separator between hedge and sentence
            hedge_clean = hedge.rstrip()
            if hedge_clean.endswith("that") or hedge_clean.endswith("which"):
                # "that"/"which" hedges connect directly without comma
                hedged = f"{hedge[0].upper()}{hedge[1:]} {sent_modified}"
            elif hedge_clean.endswith("argue") or hedge_clean.endswith("say"):
                # Hedges like "one could argue" need "that" before the clause
                hedged = f"{hedge[0].upper()}{hedge[1:]} that {sent_modified}"
            else:
                hedged = f"{hedge[0].upper()}{hedge[1:]}, {sent_modified}"
            result_sents.append(hedged)
        else:
            result_sents.append(sent)

    return " ".join(result_sents)


def _inject_personal_markers(
    text: str, domain: WritingDomain, rng: random.Random, creativity: float
) -> str:
    """Add first-person experience markers to some paragraphs."""
    markers = PERSONAL_MARKERS.get(domain, PERSONAL_MARKERS[WritingDomain.CASUAL])
    paragraphs = text.split("\n\n")
    if len(paragraphs) < 2:
        return text

    inject_rate = 0.2 + creativity * 0.15

    result_paras = []
    for i, para in enumerate(paragraphs):
        if i > 0 and rng.random() < inject_rate and len(para.split()) > 10:
            marker = rng.choice(markers)
            sentences = re.split(r'(?<=[.!?])\s+', para.strip())
            if len(sentences) > 1:
                # Insert marker at start of a random (non-first) sentence
                idx = rng.randint(0, min(1, len(sentences) - 1))
                sent = sentences[idx]
                sent_lower = sent[0].lower() + sent[1:] if sent[0].isupper() else sent
                sentences[idx] = f"{marker} {sent_lower}"
                result_paras.append(" ".join(sentences))
            else:
                result_paras.append(para)
        else:
            result_paras.append(para)

    return "\n\n".join(result_paras)


def _inject_asides(text: str, domain: WritingDomain, rng: random.Random, creativity: float) -> str:
    """Insert parenthetical asides into longer sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    inject_rate = 0.10 + creativity * 0.10
    domain_asides = ASIDES.get(domain, ASIDES[WritingDomain.CASUAL])

    result_sents = []
    for sent in sentences:
        words = sent.split()
        # Skip if sentence already contains parenthetical/aside content
        has_aside = "—" in sent or "(" in sent or ", of course," in sent.lower()
        if len(words) > 12 and rng.random() < inject_rate and not has_aside:
            aside = rng.choice(domain_asides)
            # Insert aside after roughly the midpoint
            mid = len(words) // 2
            # Find a natural break point (after a comma or conjunction)
            insert_pos = mid
            for offset in range(3):
                for pos in [mid + offset, mid - offset]:
                    if 0 < pos < len(words):
                        w = words[pos].rstrip(",;")
                        if w.lower() in {"and", "but", "or", "which", "that", "so", "because"}:
                            insert_pos = pos + 1
                            break
                        if words[pos].endswith(","):
                            insert_pos = pos + 1
                            break

            words.insert(insert_pos, aside)
            result_sents.append(" ".join(words))
        else:
            result_sents.append(sent)

    return " ".join(result_sents)


def _inject_opinions(
    text: str, domain: WritingDomain, rng: random.Random, creativity: float
) -> str:
    """Add opinion/judgment markers to some paragraph starts."""
    markers = OPINION_MARKERS.get(domain, OPINION_MARKERS[WritingDomain.CASUAL])
    paragraphs = text.split("\n\n")
    if len(paragraphs) < 3:
        return text

    # Only inject opinion into 1 paragraph max
    inject_rate = 0.15 + creativity * 0.10
    injected = False

    result_paras = []
    for i, para in enumerate(paragraphs):
        if (
            not injected
            and i >= 1
            and rng.random() < inject_rate
            and len(para.split()) > 15
        ):
            marker = rng.choice(markers)
            sentences = re.split(r'(?<=[.!?])\s+', para.strip())
            if sentences:
                # Prepend opinion to first sentence of paragraph
                sent = sentences[0]
                sent_lower = sent[0].lower() + sent[1:] if sent[0].isupper() else sent
                sentences[0] = f"{marker} {sent_lower}"
                result_paras.append(" ".join(sentences))
                injected = True
            else:
                result_paras.append(para)
        else:
            result_paras.append(para)

    return "\n\n".join(result_paras)
