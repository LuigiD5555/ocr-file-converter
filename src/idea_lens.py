"""IdeaLens: lightweight important-idea detection for normalized documents.

IdeaLens is intentionally local-first and dependency-light. It does not use
LLMs, embeddings, or a vector database. It combines frequency, phrase
co-occurrence, document position, and relation markers to extract the ideas
that best preserve a document's conceptual structure.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field


DEFAULT_STOPWORDS = {
    # English
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "his", "how", "i", "if", "in", "into", "is", "it", "its", "may",
    "more", "most", "not", "of", "on", "or", "our", "she", "should", "so",
    "such", "than", "that", "the", "their", "then", "there", "these", "they",
    "this", "those", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "will", "with", "would", "you", "your",
    # Spanish
    "al", "algo", "ante", "antes", "aqui", "aquí", "como", "con", "contra",
    "cual", "cuál", "cuando", "cuándo", "de", "del", "desde", "donde",
    "dónde", "dos", "el", "él", "ella", "ellas", "ellos", "en", "entre", "era",
    "eran", "es", "esa", "esas", "ese", "eso", "esos", "esta", "está", "estas",
    "este", "esto", "estos", "fue", "han", "hay", "la", "las", "le", "les",
    "lo", "los", "mas", "más", "mi", "mis", "muy", "no", "nos", "o", "para",
    "pero", "por", "porque", "que", "qué", "se", "ser", "si", "sí", "sin",
    "sobre", "son", "su", "sus", "tambien", "también", "te", "un", "una",
    "uno", "unos", "y", "ya",
}

RELATION_MARKERS: Dict[str, Tuple[str, ...]] = {
    "definition": (
        "se define", "se refiere", "consiste en", "significa", "es decir",
        "is defined", "refers to", "consists of", "means", "that is",
    ),
    "contrast": (
        "sin embargo", "aunque", "a diferencia", "no obstante", "en cambio",
        "however", "although", "whereas", "unlike", "in contrast",
    ),
    "cause_effect": (
        "por lo tanto", "por tanto", "debido a", "provoca", "permite", "impide",
        "therefore", "because", "due to", "causes", "enables", "prevents",
    ),
    "rule": (
        "debe", "requiere", "se requiere", "obligatorio", "criterio", "condición",
        "must", "requires", "shall", "criterion", "condition",
    ),
    "exception": (
        "excepto", "salvo", "a menos que", "no aplica", "unless", "except",
    ),
    "example": (
        "por ejemplo", "ejemplo", "caso", "for example", "example", "case",
    ),
    "conclusion": (
        "en conclusión", "conclusión", "finalmente", "therefore", "in conclusion",
    ),
}


class KeywordSignal(BaseModel):
    """A single keyword or keyphrase candidate."""

    text: str
    score: float
    occurrences: int
    first_position: float = Field(..., ge=0.0, le=1.0)
    kind: str = Field(default="keyword", description="keyword or keyphrase")


class IdeaAnchor(BaseModel):
    """A source sentence or short passage that preserves a high-value idea."""

    id: str
    text: str
    score: float
    kind: str
    position: float = Field(..., ge=0.0, le=1.0)
    keywords: List[str] = Field(default_factory=list)
    why_important: List[str] = Field(default_factory=list)


class IdeaLensReport(BaseModel):
    """Structured important-idea report for one normalized text."""

    source: str
    language: str = "auto"
    text_length: int
    sentence_count: int
    keywords: List[KeywordSignal]
    keyphrases: List[KeywordSignal]
    anchors: List[IdeaAnchor]
    outline: List[str] = Field(default_factory=list)
    algorithm: str = "idealens.statistical_graph.v1"


@dataclass(frozen=True)
class IdeaLensConfig:
    """Runtime configuration for IdeaLens."""

    keyword_limit: int = 50
    keyphrase_limit: int = 50
    anchor_limit: int = 20
    min_token_len: int = 3
    max_ngram: int = 4
    sentence_window: int = 4
    stopwords: Optional[Sequence[str]] = None


class IdeaLens:
    """Detect keywords, keyphrases and idea anchors from normalized text."""

    def __init__(self, config: Optional[IdeaLensConfig] = None) -> None:
        self.config = config or IdeaLensConfig()
        self.stopwords = set(DEFAULT_STOPWORDS)
        if self.config.stopwords:
            self.stopwords.update(token.lower() for token in self.config.stopwords)

    def analyze(self, text: str, *, source: str = "", language: str = "auto") -> IdeaLensReport:
        cleaned = _normalize_for_analysis(text)
        sentences = _split_sentences(cleaned)
        tokenized = [self._tokens(sentence) for sentence in sentences]

        term_counts, first_seen = self._count_terms(tokenized)
        graph_degree = self._cooccurrence_degree(tokenized)
        keyword_signals = self._rank_keywords(term_counts, first_seen, graph_degree)
        phrase_counts, phrase_first_seen = self._count_phrases(tokenized)
        phrase_signals = self._rank_phrases(phrase_counts, phrase_first_seen, graph_degree)
        anchors = self._rank_anchors(sentences, tokenized, keyword_signals, phrase_signals)
        outline = self._build_outline(anchors)

        return IdeaLensReport(
            source=source,
            language=language,
            text_length=len(text),
            sentence_count=len(sentences),
            keywords=keyword_signals[: self.config.keyword_limit],
            keyphrases=phrase_signals[: self.config.keyphrase_limit],
            anchors=anchors[: self.config.anchor_limit],
            outline=outline,
        )

    def save_report(self, report: IdeaLensReport, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
        return output_path

    def _tokens(self, text: str) -> List[str]:
        tokens = re.findall(r"[\wÁÉÍÓÚÜÑáéíóúüñ][\wÁÉÍÓÚÜÑáéíóúüñ_-]*", text.lower())
        result: List[str] = []
        for token in tokens:
            token = token.strip("_-.")
            if len(token) < self.config.min_token_len:
                continue
            if token.isdigit():
                continue
            if token in self.stopwords:
                continue
            result.append(token)
        return result

    def _count_terms(self, tokenized: Sequence[Sequence[str]]) -> Tuple[Counter[str], Dict[str, int]]:
        counts: Counter[str] = Counter()
        first_seen: Dict[str, int] = {}
        for sentence_index, tokens in enumerate(tokenized):
            for token in tokens:
                counts[token] += 1
                first_seen.setdefault(token, sentence_index)
        return counts, first_seen

    def _count_phrases(self, tokenized: Sequence[Sequence[str]]) -> Tuple[Counter[str], Dict[str, int]]:
        counts: Counter[str] = Counter()
        first_seen: Dict[str, int] = {}
        for sentence_index, tokens in enumerate(tokenized):
            if len(tokens) < 2:
                continue
            max_ngram = min(self.config.max_ngram, len(tokens))
            for size in range(2, max_ngram + 1):
                for index in range(0, len(tokens) - size + 1):
                    phrase_tokens = tokens[index : index + size]
                    if len(set(phrase_tokens)) == 1:
                        continue
                    phrase = " ".join(phrase_tokens)
                    counts[phrase] += 1
                    first_seen.setdefault(phrase, sentence_index)
        return counts, first_seen

    def _cooccurrence_degree(self, tokenized: Sequence[Sequence[str]]) -> Counter[str]:
        degree: Counter[str] = Counter()
        window = max(2, self.config.sentence_window)
        for tokens in tokenized:
            unique_tokens = list(dict.fromkeys(tokens))
            for index, token in enumerate(unique_tokens):
                neighbors = unique_tokens[max(0, index - window) : index] + unique_tokens[index + 1 : index + 1 + window]
                degree[token] += len(set(neighbors))
        return degree

    def _rank_keywords(
        self,
        counts: Counter[str],
        first_seen: Dict[str, int],
        graph_degree: Counter[str],
    ) -> List[KeywordSignal]:
        total_positions = max(first_seen.values(), default=0) + 1
        signals: List[KeywordSignal] = []
        for token, occurrences in counts.items():
            if occurrences < 2 and graph_degree[token] < 2:
                continue
            position = first_seen.get(token, 0) / max(total_positions, 1)
            early_boost = 1.0 + max(0.0, 0.25 - position) * 1.2
            score = (math.log1p(occurrences) * 2.0 + math.log1p(graph_degree[token])) * early_boost
            signals.append(
                KeywordSignal(
                    text=token,
                    score=round(score, 4),
                    occurrences=occurrences,
                    first_position=round(position, 4),
                    kind="keyword",
                )
            )
        signals.sort(key=lambda item: (-item.score, item.first_position, item.text))
        return signals

    def _rank_phrases(
        self,
        counts: Counter[str],
        first_seen: Dict[str, int],
        graph_degree: Counter[str],
    ) -> List[KeywordSignal]:
        total_positions = max(first_seen.values(), default=0) + 1
        signals: List[KeywordSignal] = []
        for phrase, occurrences in counts.items():
            words = phrase.split()
            if occurrences < 2 and len(words) < 3:
                continue
            phrase_degree = sum(graph_degree[word] for word in words)
            position = first_seen.get(phrase, 0) / max(total_positions, 1)
            length_boost = 1.0 + min(len(words), self.config.max_ngram) * 0.18
            early_boost = 1.0 + max(0.0, 0.2 - position)
            score = (math.log1p(occurrences) * 2.5 + math.log1p(phrase_degree)) * length_boost * early_boost
            signals.append(
                KeywordSignal(
                    text=phrase,
                    score=round(score, 4),
                    occurrences=occurrences,
                    first_position=round(position, 4),
                    kind="keyphrase",
                )
            )
        signals.sort(key=lambda item: (-item.score, item.first_position, item.text))
        return signals

    def _rank_anchors(
        self,
        sentences: Sequence[str],
        tokenized: Sequence[Sequence[str]],
        keywords: Sequence[KeywordSignal],
        keyphrases: Sequence[KeywordSignal],
    ) -> List[IdeaAnchor]:
        keyword_weights = {item.text: item.score for item in keywords[: max(self.config.keyword_limit, 25)]}
        phrase_weights = {item.text: item.score for item in keyphrases[: max(self.config.keyphrase_limit, 25)]}
        anchors: List[IdeaAnchor] = []
        total = max(len(sentences), 1)
        for index, sentence in enumerate(sentences):
            if len(sentence.split()) < 5:
                continue
            tokens = tokenized[index] if index < len(tokenized) else []
            if not tokens:
                continue
            lowered = sentence.lower()
            token_score = sum(keyword_weights.get(token, 0.0) for token in set(tokens))
            phrase_hits = [phrase for phrase in phrase_weights if phrase in lowered]
            phrase_score = sum(phrase_weights[phrase] for phrase in phrase_hits)
            relation_kind, relation_reasons, relation_score = _relation_score(lowered)
            position = index / total
            structural_boost = 1.15 if position < 0.12 else 1.0
            density = (token_score + phrase_score) / max(len(tokens), 1)
            score = density * structural_boost + relation_score
            if score <= 0:
                continue
            hit_keywords = sorted(set(tokens), key=lambda token: -keyword_weights.get(token, 0.0))[:8]
            reasons = []
            if hit_keywords:
                reasons.append("contains high-ranking document terms")
            if phrase_hits:
                reasons.append("contains high-ranking keyphrases")
            reasons.extend(relation_reasons)
            if position < 0.12:
                reasons.append("appears early in the document")
            anchors.append(
                IdeaAnchor(
                    id=f"idea_{len(anchors) + 1:04d}",
                    text=sentence.strip(),
                    score=round(score, 4),
                    kind=relation_kind,
                    position=round(position, 4),
                    keywords=hit_keywords,
                    why_important=reasons[:5],
                )
            )
        anchors.sort(key=lambda item: (-item.score, item.position))
        return anchors

    def _build_outline(self, anchors: Sequence[IdeaAnchor]) -> List[str]:
        selected = sorted(anchors[: self.config.anchor_limit], key=lambda item: item.position)
        return [anchor.text for anchor in selected[: min(12, len(selected))]]


def save_idea_report(report: IdeaLensReport, output_path: Path) -> Path:
    """Persist an IdeaLens report as pretty JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    return output_path


def load_stopwords(path: Optional[Path]) -> List[str]:
    """Load one stopword per line. Empty lines and # comments are ignored."""

    if not path:
        return []
    words: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        words.append(stripped.lower())
    return words


def _normalize_for_analysis(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts: List[str] = []
    for paragraph in re.split(r"\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph.split()) <= 18 and not paragraph.endswith((".", ":", ";", "?", "!")):
            parts.append(paragraph)
            continue
        chunks = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÜÑ0-9])", paragraph)
        parts.extend(chunk.strip() for chunk in chunks if chunk.strip())
    return parts


def _relation_score(sentence: str) -> Tuple[str, List[str], float]:
    for kind, markers in RELATION_MARKERS.items():
        for marker in markers:
            if marker in sentence:
                return kind, [f"matches {kind.replace('_', '-')} language"], 1.8
    return "concept", [], 0.0
