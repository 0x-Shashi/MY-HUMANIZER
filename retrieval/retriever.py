"""
RAG retriever for human writing style grounding.

Inspired by 2404.08189v1 (Béchard & Marquez Ayala, NAACL 2024):
- Uses a dense retriever to find similar human-written text
- Retrieved samples provide structural/stylistic templates
- Grounds transformations in REAL human writing patterns

Architecture:
1. Corpus of human-written text (user-provided or bootstrapped)
2. Sentence-level embeddings using sentence-transformers
3. FAISS index for fast cosine similarity retrieval
4. Top-K retrieval returns human paragraphs similar to input

The key insight from the paper:
- A small 110M-param retriever + LLM outperforms 15.5B-param LLM alone
- Retrieval reduces hallucination (in our case: AI fingerprints) by 3-5×
- Separate training of retriever and generator is practical

For our humanizer, RAG provides:
- Real human sentence structures to mimic
- Natural transition patterns from actual writing
- Vocabulary diversity from authentic human text
- Burstiness templates (sentence length patterns)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

CORPUS_DIR = Path(__file__).parent / "corpus"


@dataclass
class CorpusDocument:
    """A document in the human writing corpus."""
    text: str
    source: str = ""
    domain: str = ""
    sentences: list[str] = field(default_factory=list)


@dataclass
class RetrievedSample:
    """A retrieved human writing sample."""
    text: str
    similarity: float
    source: str = ""
    sentence_lengths: list[int] = field(default_factory=list)
    transitions: list[str] = field(default_factory=list)


class HumanCorpusManager:
    """
    Manages the human writing corpus for RAG retrieval.

    Supports two modes:
    1. Basic (no ML dependencies): TF-IDF based retrieval
    2. Advanced (with sentence-transformers): Dense embedding retrieval
    """

    def __init__(self, corpus_dir: Optional[Path] = None):
        self.corpus_dir = corpus_dir or CORPUS_DIR
        self.documents: list[CorpusDocument] = []
        self._embeddings = None
        self._index = None
        self._encoder = None

    def load_corpus(self) -> int:
        """Load all documents from corpus directory. Returns count loaded."""
        self.documents = []

        if not self.corpus_dir.exists():
            self.corpus_dir.mkdir(parents=True, exist_ok=True)
            return 0

        for filepath in self.corpus_dir.iterdir():
            if filepath.suffix == ".txt":
                text = filepath.read_text(encoding="utf-8", errors="ignore")
                sentences = _split_sentences(text)
                self.documents.append(CorpusDocument(
                    text=text,
                    source=filepath.name,
                    sentences=sentences,
                ))
            elif filepath.suffix == ".json":
                data = json.loads(filepath.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        text = item.get("text", "")
                        sentences = _split_sentences(text)
                        self.documents.append(CorpusDocument(
                            text=text,
                            source=item.get("source", filepath.name),
                            domain=item.get("domain", ""),
                            sentences=sentences,
                        ))

        return len(self.documents)

    def add_document(self, text: str, source: str = "", domain: str = "") -> None:
        """Add a document to the corpus."""
        sentences = _split_sentences(text)
        self.documents.append(CorpusDocument(
            text=text, source=source, domain=domain, sentences=sentences,
        ))

    def save_document(self, text: str, filename: str) -> Path:
        """Save a document to the corpus directory."""
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.corpus_dir / filename
        filepath.write_text(text, encoding="utf-8")
        self.add_document(text, source=filename)
        return filepath

    def retrieve_similar(
        self, query: str, top_k: int = 3
    ) -> list[RetrievedSample]:
        """
        Retrieve the most similar human-written samples to the query.

        Uses TF-IDF cosine similarity (basic mode) or dense embeddings (advanced mode).
        """
        if not self.documents:
            return []

        # Try advanced mode (sentence-transformers + FAISS)
        if self._try_init_encoder():
            return self._retrieve_dense(query, top_k)

        # Fallback: TF-IDF based retrieval
        return self._retrieve_tfidf(query, top_k)

    def _try_init_encoder(self) -> bool:
        """Try to initialize sentence-transformers encoder."""
        if self._encoder is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            return True
        except ImportError:
            return False

    def _retrieve_dense(self, query: str, top_k: int) -> list[RetrievedSample]:
        """Dense retrieval using sentence-transformers."""
        if self._encoder is None:
            return []

        import numpy as np

        # Encode query
        query_emb = self._encoder.encode([query], normalize_embeddings=True)

        # Encode corpus (cached)
        if self._embeddings is None:
            texts = [doc.text[:512] for doc in self.documents]
            self._embeddings = self._encoder.encode(texts, normalize_embeddings=True)

        # Cosine similarity (embeddings are normalized)
        similarities = np.dot(self._embeddings, query_emb.T).flatten()

        # Top-K
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            sim = float(similarities[idx])
            if sim < 0.1:
                continue

            sent_lengths = [len(s.split()) for s in doc.sentences]
            transitions = _extract_transitions(doc.text)

            results.append(RetrievedSample(
                text=doc.text[:500],
                similarity=sim,
                source=doc.source,
                sentence_lengths=sent_lengths,
                transitions=transitions,
            ))

        return results

    def _retrieve_tfidf(self, query: str, top_k: int) -> list[RetrievedSample]:
        """Basic TF-IDF retrieval (no ML dependencies)."""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))

        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self.documents):
            doc_words = set(re.findall(r'\b\w+\b', doc.text.lower()))
            if not doc_words:
                continue
            # Jaccard similarity
            intersection = query_words & doc_words
            union = query_words | doc_words
            score = len(intersection) / len(union) if union else 0.0
            scored.append((score, i))

        scored.sort(reverse=True)

        results = []
        for score, idx in scored[:top_k]:
            doc = self.documents[idx]
            if score < 0.05:
                continue

            sent_lengths = [len(s.split()) for s in doc.sentences]
            transitions = _extract_transitions(doc.text)

            results.append(RetrievedSample(
                text=doc.text[:500],
                similarity=score,
                source=doc.source,
                sentence_lengths=sent_lengths,
                transitions=transitions,
            ))

        return results

    def get_burstiness_template(self, target_length: int = 10) -> list[int]:
        """
        Get a sentence-length pattern from a random human document.
        Used to guide burstiness shaping.
        """
        if not self.documents:
            # Fallback: synthetic human-like pattern
            return [8, 22, 5, 31, 12, 3, 18, 7, 25, 14]

        import random as _rng
        doc = _rng.choice(self.documents)
        lengths = [len(s.split()) for s in doc.sentences]
        if len(lengths) < target_length:
            return lengths
        start = _rng.randint(0, len(lengths) - target_length)
        return lengths[start:start + target_length]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if s.strip()]


def _extract_transitions(text: str) -> list[str]:
    """Extract transition words/phrases from text."""
    transitions = []
    text_lower = text.lower()
    patterns = [
        "however", "but", "yet", "still", "though", "although",
        "meanwhile", "then", "next", "later", "finally",
        "also", "too", "and", "plus", "besides",
    ]
    for p in patterns:
        if re.search(rf'\b{re.escape(p)}\b', text_lower):
            transitions.append(p)
    return transitions
