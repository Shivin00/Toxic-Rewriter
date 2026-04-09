from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re


COURTESY_WORDS = {
    "please",
    "thank you",
    "thanks",
    "appreciate",
    "kindly",
    "could you",
    "would you",
    "let us",
    "let's",
    "i suggest",
    "perhaps",
    "consider",
    "i understand",
    "i hear your concern",
}

HARSH_WORDS = {
    "stupid",
    "idiot",
    "dumb",
    "hate",
    "awful",
    "worst",
    "useless",
    "terrible",
    "trash",
    "shut up",
    "nonsense",
    "pathetic",
    "clueless",
}

AGGRESSIVE_PATTERNS = {
    "you always",
    "you never",
    "what is wrong with you",
    "this is ridiculous",
    "nobody asked",
    "listen carefully",
}

SOFTENERS = {
    "i think",
    "it may help",
    "might be better",
    "could be improved",
    "perhaps",
    "consider",
    "i understand",
    "i would recommend",
}

ASSERTIVE_MARKERS = {
    "need",
    "must",
    "require",
    "expect",
    "boundary",
    "issue",
    "request",
    "important",
    "please address",
}

PROFESSIONAL_MARKERS = {
    "regarding",
    "appreciate",
    "support",
    "recommend",
    "request",
    "clarify",
    "review",
    "concern",
    "resolution",
}


@dataclass(slots=True)
class ToneProfile:
    courtesy_hits: int
    harsh_hits: int
    aggressive_hits: int
    softener_hits: int
    exclamation_count: int
    uppercase_ratio: float


@dataclass(slots=True)
class EvaluationResult:
    toxicity_before: float
    toxicity_after: float
    politeness_before: float
    politeness_after: float
    semantic_similarity: float
    confidence_score: float
    tone_before: ToneProfile
    tone_after: ToneProfile
    assertiveness_before: float
    assertiveness_after: float
    empathy_after: float
    clarity_after: float
    professionalism_after: float

    @property
    def toxicity_delta(self) -> float:
        return self.toxicity_before - self.toxicity_after

    @property
    def politeness_delta(self) -> float:
        return self.politeness_after - self.politeness_before


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _count_phrase_hits(text: str, phrases: set[str]) -> int:
    normalized = _normalize(text)
    return sum(1 for phrase in phrases if phrase in normalized)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def build_tone_profile(text: str) -> ToneProfile:
    normalized = _normalize(text)
    alpha_chars = [char for char in text if char.isalpha()]
    uppercase_chars = [char for char in alpha_chars if char.isupper()]

    return ToneProfile(
        courtesy_hits=_count_phrase_hits(normalized, COURTESY_WORDS),
        harsh_hits=_count_phrase_hits(normalized, HARSH_WORDS),
        aggressive_hits=_count_phrase_hits(normalized, AGGRESSIVE_PATTERNS),
        softener_hits=_count_phrase_hits(normalized, SOFTENERS),
        exclamation_count=min(text.count("!"), 6),
        uppercase_ratio=_safe_ratio(len(uppercase_chars), len(alpha_chars)),
    )


def heuristic_politeness_score(text: str, toxicity_score: float) -> float:
    normalized = _normalize(text)
    if not normalized:
        return 0.0

    profile = build_tone_profile(text)
    courtesy_bonus = min(profile.courtesy_hits * 0.08, 0.24)
    softener_bonus = min(profile.softener_hits * 0.06, 0.18)
    harsh_penalty = min(profile.harsh_hits * 0.12, 0.42)
    aggressive_penalty = min(profile.aggressive_hits * 0.10, 0.25)
    exclamation_penalty = profile.exclamation_count * 0.03
    uppercase_penalty = min(profile.uppercase_ratio * 0.5, 0.18)

    score = (
        0.68 * (1 - toxicity_score)
        + courtesy_bonus
        + softener_bonus
        - harsh_penalty
        - aggressive_penalty
        - exclamation_penalty
        - uppercase_penalty
    )
    return max(0.0, min(score, 1.0))


def semantic_similarity_score(original_text: str, rewritten_text: str) -> float:
    original = _normalize(original_text)
    rewritten = _normalize(rewritten_text)
    if not original and not rewritten:
        return 1.0
    return SequenceMatcher(a=original, b=rewritten).ratio()


def assertiveness_score(text: str, toxicity_score: float) -> float:
    normalized = _normalize(text)
    if not normalized:
        return 0.0

    marker_hits = _count_phrase_hits(normalized, ASSERTIVE_MARKERS)
    question_penalty = 0.08 if "?" in text and marker_hits == 0 else 0.0
    harsh_penalty = min(_count_phrase_hits(normalized, HARSH_WORDS) * 0.06, 0.2)
    score = 0.35 + min(marker_hits * 0.1, 0.4) + 0.15 * (1 - toxicity_score) - question_penalty - harsh_penalty
    return max(0.0, min(score, 1.0))


def empathy_score(text: str) -> float:
    normalized = _normalize(text)
    marker_hits = _count_phrase_hits(normalized, COURTESY_WORDS | {"i understand", "i hear", "i appreciate"})
    harsh_penalty = min(_count_phrase_hits(normalized, HARSH_WORDS) * 0.1, 0.3)
    score = 0.25 + min(marker_hits * 0.12, 0.48) - harsh_penalty
    return max(0.0, min(score, 1.0))


def clarity_score(text: str) -> float:
    normalized = _normalize(text)
    if not normalized:
        return 0.0
    words = normalized.split()
    sentence_count = max(1, len(re.findall(r"[.!?]", text)))
    avg_sentence_length = len(words) / sentence_count
    length_penalty = min(max(avg_sentence_length - 18, 0) * 0.02, 0.25)
    noise_penalty = min(text.count("!") * 0.03, 0.15)
    score = 0.78 - length_penalty - noise_penalty
    return max(0.0, min(score, 1.0))


def professionalism_score(text: str) -> float:
    normalized = _normalize(text)
    marker_hits = _count_phrase_hits(normalized, PROFESSIONAL_MARKERS)
    harsh_penalty = min(_count_phrase_hits(normalized, HARSH_WORDS) * 0.12, 0.5)
    slang_penalty = 0.12 if any(token in normalized for token in {"bro", "wtf", "lol", "nah"}) else 0.0
    score = 0.45 + min(marker_hits * 0.08, 0.32) - harsh_penalty - slang_penalty
    return max(0.0, min(score, 1.0))


def build_evaluation_result(
    original_text: str,
    rewritten_text: str,
    toxicity_before: float,
    toxicity_after: float,
) -> EvaluationResult:
    politeness_before = heuristic_politeness_score(original_text, toxicity_before)
    politeness_after = heuristic_politeness_score(rewritten_text, toxicity_after)
    semantic_similarity = semantic_similarity_score(original_text, rewritten_text)
    tone_before = build_tone_profile(original_text)
    tone_after = build_tone_profile(rewritten_text)
    assertiveness_before = assertiveness_score(original_text, toxicity_before)
    assertiveness_after = assertiveness_score(rewritten_text, toxicity_after)
    empathy_after_value = empathy_score(rewritten_text)
    clarity_after_value = clarity_score(rewritten_text)
    professionalism_after_value = professionalism_score(rewritten_text)

    confidence_score = (
        0.35 * max(0.0, toxicity_before - toxicity_after)
        + 0.25 * max(0.0, politeness_after - politeness_before)
        + 0.15 * semantic_similarity
        + 0.10 * empathy_after_value
        + 0.15 * professionalism_after_value
    )

    return EvaluationResult(
        toxicity_before=toxicity_before,
        toxicity_after=toxicity_after,
        politeness_before=politeness_before,
        politeness_after=politeness_after,
        semantic_similarity=semantic_similarity,
        confidence_score=max(0.0, min(confidence_score, 1.0)),
        tone_before=tone_before,
        tone_after=tone_after,
        assertiveness_before=assertiveness_before,
        assertiveness_after=assertiveness_after,
        empathy_after=empathy_after_value,
        clarity_after=clarity_after_value,
        professionalism_after=professionalism_after_value,
    )
