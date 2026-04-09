from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any

from toxic_rewriter.scoring import EvaluationResult, build_evaluation_result


DIMENSION_PATTERNS = {
    "hate_speech": {
        "slur",
        "go back",
        "these people",
        "your kind",
        "inferior",
    },
    "harassment": {
        "idiot",
        "stupid",
        "clown",
        "moron",
        "you are useless",
        "pathetic",
    },
    "threats": {
        "i will ruin",
        "i will destroy",
        "watch your back",
        "you will regret",
        "hurt you",
        "fire you",
    },
    "profanity": {
        "damn",
        "hell",
        "wtf",
        "trash",
        "crap",
    },
    "discrimination": {
        "your people",
        "girls can't",
        "boys can't",
        "foreigners",
        "those people",
    },
    "sarcasm": {
        "yeah right",
        "obviously",
        "sure you did",
        "great job",
        "brilliant idea",
    },
    "gaslighting": {
        "you're imagining things",
        "that never happened",
        "you're too sensitive",
        "you're overreacting",
        "you're crazy",
    },
    "microaggressions": {
        "you speak well for",
        "where are you really from",
        "articulate for",
        "you people",
        "surprisingly smart",
    },
}

HIDDEN_TOXICITY_PATTERNS = {
    "passive_aggression": {
        "i'm not saying",
        "no offense but",
        "as usual",
        "if you had listened",
        "whatever you say",
    },
    "manipulative_kindness": {
        "with all due respect",
        "i only want what's best for you",
        "be reasonable",
        "just trying to help",
    },
    "condescension": {
        "let me explain this simply",
        "it's not that hard",
        "even you should understand",
        "basic stuff",
    },
}

TOXIC_HINTS = {
    "idiot": "there may be a misunderstanding",
    "stupid": "this could be improved",
    "dumb": "this may not be the best approach",
    "hate": "strongly disagree with",
    "shut up": "please let me finish",
    "worst": "least effective",
    "awful": "not ideal",
    "useless": "not very helpful",
    "trash": "low quality",
    "pathetic": "disappointing",
    "clueless": "missing context",
    "terrible": "frustrating",
    "bad": "not working well",
}

STYLE_GUIDELINES = {
    "formal professional": "Use a formal, professional, and respectful tone.",
    "casual friendly": "Use a warm, friendly, and natural conversational tone.",
    "empathetic supportive": "Use an empathetic, supportive, and validating tone.",
    "assertive non-toxic": "Keep the message firm and assertive without hostility.",
    "therapeutic": "Use calm, reflective, non-judgmental mental-health communication norms.",
    "diplomatic corporate": "Use diplomatic, corporate-safe language suitable for workplace escalation.",
    "neutral factual": "Use neutral, concise, fact-based wording with minimal emotion.",
}

STYLE_PREFIX = {
    "formal professional": "I would like to note that",
    "casual friendly": "I think",
    "empathetic supportive": "I understand the concern, and",
    "assertive non-toxic": "I need to be clear that",
    "therapeutic": "I want to express this calmly:",
    "diplomatic corporate": "From a professional standpoint,",
    "neutral factual": "To state the issue directly,",
}

PERSONA_GUIDANCE = {
    "boss": "Speak respectfully and solution-first.",
    "friend": "Keep it honest but warm.",
    "customer": "Be service-oriented and calm.",
    "coworker": "Stay direct, collaborative, and professional.",
    "partner": "Use emotionally aware, non-accusatory language.",
    "public audience": "Keep the tone broad, safe, and reputation-aware.",
}


@dataclass(slots=True)
class ReplacementRecord:
    source: str
    replacement: str
    category: str


@dataclass(slots=True)
class ToxicityDimensionScore:
    name: str
    score: float
    evidence: list[str]


@dataclass(slots=True)
class IntentProfile:
    original_intent: str
    goal: str
    target_entity: str
    emotion: str
    strength: float


@dataclass(slots=True)
class SafetyAssessment:
    overall_risk: float
    flagged: bool
    reasons: list[str]


@dataclass(slots=True)
class RewriteAnalysis:
    rewritten_text: str
    evaluation: EvaluationResult
    notes: list[str]
    replacements: list[ReplacementRecord]
    style: str
    original_dimensions: list[ToxicityDimensionScore]
    rewritten_dimensions: list[ToxicityDimensionScore]
    hidden_toxicity: list[ToxicityDimensionScore]
    safety: SafetyAssessment
    intent: IntentProfile
    flow_steps: list[dict[str, str]]


@dataclass(slots=True)
class ToxicRewriterService:
    classifier: Any = None
    rewriter: Any = None
    classification_backend: str = "heuristic"
    rewrite_backend: str = "heuristic"

    def __post_init__(self) -> None:
        self._load_classifier()
        self._load_rewriter()

    def _load_classifier(self) -> None:
        try:
            from transformers import pipeline

            self.classifier = pipeline(
                task="text-classification",
                model="unitary/toxic-bert",
                top_k=None,
            )
            self.classification_backend = "unitary/toxic-bert"
        except Exception:
            self.classifier = None
            self.classification_backend = "heuristic"

    def _load_rewriter(self) -> None:
        try:
            from transformers import pipeline

            self.rewriter = pipeline(
                task="text2text-generation",
                model="google/flan-t5-base",
            )
            self.rewrite_backend = "google/flan-t5-base"
        except Exception:
            self.rewriter = None
            self.rewrite_backend = "heuristic"

    def classify_toxicity(self, text: str) -> float:
        if not text.strip():
            return 0.0

        if self.classifier is not None:
            try:
                raw_predictions = self.classifier(text)
                predictions = (
                    raw_predictions[0]
                    if raw_predictions and isinstance(raw_predictions[0], list)
                    else raw_predictions
                )
                label_map = {
                    item["label"].lower(): float(item["score"]) for item in predictions
                }
                if "toxic" in label_map:
                    return max(0.0, min(label_map["toxic"], 1.0))
            except Exception:
                pass

        dimension_scores = self.classify_dimensions(text)
        hidden_scores = self.detect_hidden_toxicity(text)
        max_dimension = max((item.score for item in dimension_scores), default=0.0)
        hidden_boost = max((item.score for item in hidden_scores), default=0.0) * 0.4
        punctuation_boost = min(text.count("!"), 5) * 0.04
        all_caps_boost = 0.1 if any(token.isupper() and len(token) > 2 for token in text.split()) else 0.0
        return max(0.0, min(max_dimension + hidden_boost + punctuation_boost + all_caps_boost, 1.0))

    def classify_dimensions(self, text: str) -> list[ToxicityDimensionScore]:
        normalized = text.lower()
        results: list[ToxicityDimensionScore] = []

        for dimension, phrases in DIMENSION_PATTERNS.items():
            evidence = [phrase for phrase in phrases if phrase in normalized]
            score = min(len(evidence) * 0.28, 1.0)
            if dimension == "profanity" and any(symbol in normalized for symbol in {"wtf", "damn"}):
                score = max(score, 0.35)
            if dimension == "threats" and any(term in normalized for term in {"ruin", "destroy", "regret"}):
                score = max(score, 0.52)
            results.append(
                ToxicityDimensionScore(
                    name=dimension.replace("_", " ").title(),
                    score=score,
                    evidence=evidence[:4],
                )
            )
        return results

    def detect_hidden_toxicity(self, text: str) -> list[ToxicityDimensionScore]:
        normalized = text.lower()
        results: list[ToxicityDimensionScore] = []
        for category, phrases in HIDDEN_TOXICITY_PATTERNS.items():
            evidence = [phrase for phrase in phrases if phrase in normalized]
            score = min(len(evidence) * 0.32, 1.0)
            results.append(
                ToxicityDimensionScore(
                    name=category.replace("_", " ").title(),
                    score=score,
                    evidence=evidence[:4],
                )
            )
        return results

    def infer_intent(
        self,
        text: str,
        context: str = "",
        emotion_target: str = "calm",
        strength: float = 0.55,
    ) -> IntentProfile:
        normalized = text.lower()
        context_lower = context.lower()

        if any(term in normalized for term in {"fix", "issue", "problem", "service", "support"}):
            goal = "complaint"
        elif any(term in normalized for term in {"disagree", "wrong", "bad idea", "not effective"}):
            goal = "disagreement"
        elif any(term in normalized for term in {"stop", "don't", "enough", "boundary"}):
            goal = "boundary-setting"
        else:
            goal = "feedback"

        target_entity = "the other person"
        if any(term in context_lower for term in {"manager", "boss"}):
            target_entity = "manager"
        elif "customer" in context_lower:
            target_entity = "customer"
        elif "friend" in context_lower:
            target_entity = "friend"

        emotion = "anger" if any(term in normalized for term in {"hate", "worst", "awful", "terrible"}) else "frustration"
        original_intent = f"{goal} with a desire to be understood while preserving firmness at {strength:.0%} intensity"

        return IntentProfile(
            original_intent=original_intent,
            goal=goal,
            target_entity=target_entity,
            emotion=emotion_target if emotion_target else emotion,
            strength=strength,
        )

    def assess_safety(
        self,
        text: str,
        rewritten_text: str,
        dimensions: list[ToxicityDimensionScore],
        hidden_toxicity: list[ToxicityDimensionScore],
    ) -> SafetyAssessment:
        reasons: list[str] = []
        dimension_lookup = {item.name: item.score for item in dimensions}
        hidden_lookup = {item.name: item.score for item in hidden_toxicity}

        risk = 0.0
        if dimension_lookup.get("Threats", 0.0) > 0.4:
            reasons.append("The original text contains threat-like language.")
            risk += 0.35
        if hidden_lookup.get("Manipulative Kindness", 0.0) > 0.3:
            reasons.append("Possible manipulative politeness was detected.")
            risk += 0.22
        if any(term in rewritten_text.lower() for term in {"destroy", "regret", "punish"}):
            reasons.append("The rewritten output still contains potentially harmful intent markers.")
            risk += 0.28
        if any(term in rewritten_text.lower() for term in {"no offense", "with all due respect"}):
            reasons.append("The rewritten output may still hide hostility behind polite framing.")
            risk += 0.20

        return SafetyAssessment(
            overall_risk=max(0.0, min(risk, 1.0)),
            flagged=risk >= 0.35,
            reasons=reasons or ["No high-risk abuse pattern was detected by the heuristic safety layer."],
        )

    def rewrite_text(
        self,
        text: str,
        style: str = "formal professional",
        context: str = "",
        persona: str = "coworker",
        emotion_target: str = "calm",
        strength: float = 0.55,
    ) -> str:
        if not text.strip():
            return ""

        normalized_style = style if style in STYLE_GUIDELINES else "formal professional"
        persona_note = PERSONA_GUIDANCE.get(persona, PERSONA_GUIDANCE["coworker"])
        prompt = (
            f"{STYLE_GUIDELINES[normalized_style]} Preserve the intent and keep firmness near {strength:.0%}. "
            f"Persona target: {persona}. {persona_note} Desired emotional shift: {emotion_target}. "
            f"Conversation context: {context or 'No extra context provided.'} "
            "Remove toxicity, threats, manipulation, and passive aggression. Return only the rewritten text."
        )

        if self.rewriter is not None:
            try:
                output = self.rewriter(
                    f"{prompt}\n\n{text}",
                    max_length=320,
                    clean_up_tokenization_spaces=True,
                )
                generated_text = output[0]["generated_text"].strip()
                if generated_text:
                    return self._post_process_rewrite(generated_text)
            except Exception:
                pass

        return self._heuristic_rewrite(
            text=text,
            style=normalized_style,
            persona=persona,
            emotion_target=emotion_target,
            strength=strength,
            context=context,
        )

    def analyze_rewrite(
        self,
        original_text: str,
        style: str = "formal professional",
        context: str = "",
        persona: str = "coworker",
        emotion_target: str = "calm",
        strength: float = 0.55,
    ) -> RewriteAnalysis:
        rewritten_text = self.rewrite_text(
            original_text,
            style=style,
            context=context,
            persona=persona,
            emotion_target=emotion_target,
            strength=strength,
        )
        original_dimensions = self.classify_dimensions(original_text)
        rewritten_dimensions = self.classify_dimensions(rewritten_text)
        hidden_toxicity = self.detect_hidden_toxicity(original_text)
        evaluation = self.evaluate_transformation(original_text, rewritten_text)
        replacements = self.extract_replacements(original_text, rewritten_text)
        intent = self.infer_intent(
            original_text,
            context=context,
            emotion_target=emotion_target,
            strength=strength,
        )
        safety = self.assess_safety(
            original_text,
            rewritten_text,
            original_dimensions,
            hidden_toxicity,
        )
        notes = self.explain_rewrite(
            original_text=original_text,
            rewritten_text=rewritten_text,
            replacements=replacements,
            dimensions=original_dimensions,
            safety=safety,
            intent=intent,
        )
        flow_steps = self.build_flow_steps(
            original_text=original_text,
            rewritten_text=rewritten_text,
            dimensions=original_dimensions,
            intent=intent,
            safety=safety,
            style=style,
        )

        return RewriteAnalysis(
            rewritten_text=rewritten_text,
            evaluation=evaluation,
            notes=notes,
            replacements=replacements,
            style=style,
            original_dimensions=original_dimensions,
            rewritten_dimensions=rewritten_dimensions,
            hidden_toxicity=hidden_toxicity,
            safety=safety,
            intent=intent,
            flow_steps=flow_steps,
        )

    def explain_rewrite(
        self,
        original_text: str,
        rewritten_text: str,
        replacements: list[ReplacementRecord],
        dimensions: list[ToxicityDimensionScore],
        safety: SafetyAssessment,
        intent: IntentProfile,
    ) -> list[str]:
        notes: list[str] = []
        if self.classify_toxicity(rewritten_text) < self.classify_toxicity(original_text):
            notes.append("The rewrite reduces overall toxicity while keeping the main complaint or boundary intact.")
        high_dimensions = [item.name for item in dimensions if item.score >= 0.25]
        if high_dimensions:
            notes.append(f"High-toxicity axes addressed: {', '.join(high_dimensions[:4])}.")
        if any(change.category == "toxic_phrase" for change in replacements):
            notes.append("Targeted toxic phrases were replaced instead of rewriting the message blindly.")
        notes.append(f"Detected goal: {intent.goal}. Target entity: {intent.target_entity}.")
        if safety.flagged:
            notes.append("A safety warning remains because intent-level risk markers were detected.")
        else:
            notes.append("The safety layer did not find a strong residual abuse signal in the rewritten output.")
        return notes

    def build_flow_steps(
        self,
        original_text: str,
        rewritten_text: str,
        dimensions: list[ToxicityDimensionScore],
        intent: IntentProfile,
        safety: SafetyAssessment,
        style: str,
    ) -> list[dict[str, str]]:
        flagged_dimensions = [item.name for item in dimensions if item.score >= 0.2]
        return [
            {
                "stage": "Detected",
                "summary": f"Flagged dimensions: {', '.join(flagged_dimensions[:5]) or 'Low explicit toxicity'}",
            },
            {
                "stage": "Interpreted",
                "summary": f"Intent inferred as {intent.goal}; target is {intent.target_entity}; desired emotion shift is {intent.emotion}.",
            },
            {
                "stage": "Rewritten",
                "summary": f"Applied {style} style while preserving roughly {intent.strength:.0%} firmness.",
            },
            {
                "stage": "Safety Check",
                "summary": "; ".join(safety.reasons[:2]),
            },
            {
                "stage": "Output",
                "summary": rewritten_text,
            },
        ]

    def extract_replacements(
        self,
        original_text: str,
        rewritten_text: str,
    ) -> list[ReplacementRecord]:
        changes: list[ReplacementRecord] = []
        original_lower = original_text.lower()

        phrase_patterns = {
            "you clearly have no idea what you're doing": "there may be some confusion around the approach",
            "this is the dumbest plan ever": "this plan may not be the most effective option",
            "what is wrong with you": "there may be a misunderstanding",
            "you never": "it would help to revisit",
            "you always": "this seems to happen often",
            "no offense but": "to be direct",
        }

        for source, replacement in phrase_patterns.items():
            if source in original_lower:
                changes.append(ReplacementRecord(source=source, replacement=replacement, category="toxic_phrase"))

        for toxic_term, neutral_phrase in TOXIC_HINTS.items():
            if re.search(rf"\b{re.escape(toxic_term)}\b", original_lower):
                changes.append(
                    ReplacementRecord(
                        source=toxic_term,
                        replacement=neutral_phrase,
                        category="toxic_phrase",
                    )
                )

        if not changes:
            matcher = SequenceMatcher(a=original_text.split(), b=rewritten_text.split())
            for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
                if opcode == "replace":
                    source = " ".join(original_text.split()[a0:a1]).strip()
                    replacement = " ".join(rewritten_text.split()[b0:b1]).strip()
                    if source and replacement:
                        changes.append(
                            ReplacementRecord(
                                source=source,
                                replacement=replacement,
                                category="style_shift",
                            )
                        )

        deduped: list[ReplacementRecord] = []
        seen: set[tuple[str, str, str]] = set()
        for change in changes:
            key = (change.source.lower(), change.replacement.lower(), change.category)
            if key not in seen:
                seen.add(key)
                deduped.append(change)
        return deduped[:10]

    def batch_rewrite(
        self,
        texts: list[str],
        style: str = "formal professional",
        context: str = "",
        persona: str = "coworker",
        emotion_target: str = "calm",
        strength: float = 0.55,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, text in enumerate(texts, start=1):
            analysis = self.analyze_rewrite(
                original_text=text,
                style=style,
                context=context,
                persona=persona,
                emotion_target=emotion_target,
                strength=strength,
            )
            eval_result = analysis.evaluation
            rows.append(
                {
                    "id": index,
                    "style": style,
                    "persona": persona,
                    "goal": analysis.intent.goal,
                    "original_text": text,
                    "rewritten_text": analysis.rewritten_text,
                    "toxicity_before": round(eval_result.toxicity_before, 4),
                    "toxicity_after": round(eval_result.toxicity_after, 4),
                    "assertiveness_after": round(eval_result.assertiveness_after, 4),
                    "empathy_after": round(eval_result.empathy_after, 4),
                    "clarity_after": round(eval_result.clarity_after, 4),
                    "professionalism_after": round(eval_result.professionalism_after, 4),
                    "safety_risk": round(analysis.safety.overall_risk, 4),
                }
            )
        return rows

    def _heuristic_rewrite(
        self,
        text: str,
        style: str,
        persona: str,
        emotion_target: str,
        strength: float,
        context: str,
    ) -> str:
        rewritten = text.strip()

        replacements = [
            (r"\byou clearly have no idea what you're doing\b", "there may be some confusion around the approach"),
            (r"\bthis is the dumbest plan ever\b", "this plan may not be the most effective option"),
            (r"\bwhat is wrong with you\b", "there may be a misunderstanding"),
            (r"\bshut up\b", "please let me finish"),
            (r"\byou are\b", "it seems"),
            (r"\byour\b", "the"),
            (r"\byou never\b", "it would help to revisit"),
            (r"\byou always\b", "this seems to happen often"),
            (r"\bstop dragging the team down\b", "please try a more coordinated approach with the team"),
            (r"\bno offense but\b", "to be direct,"),
        ]

        for pattern, replacement in replacements:
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)

        for toxic_term, neutral_phrase in TOXIC_HINTS.items():
            rewritten = re.sub(
                rf"\b{re.escape(toxic_term)}\b",
                neutral_phrase,
                rewritten,
                flags=re.IGNORECASE,
            )

        rewritten = rewritten.replace("!!", ".")
        rewritten = re.sub(r"\s+", " ", rewritten).strip(" .")

        if strength >= 0.7 and "need" not in rewritten.lower():
            rewritten = f"I need to be clear that {rewritten}"
        else:
            prefix = STYLE_PREFIX.get(style, STYLE_PREFIX["formal professional"])
            rewritten = f"{prefix} {rewritten}"

        if emotion_target == "collaborative":
            rewritten += " I would like us to work toward a practical resolution."
        elif emotion_target == "hopeful":
            rewritten += " I believe this can be improved constructively."
        elif emotion_target == "calm":
            rewritten += " I want to communicate this calmly and clearly."

        persona_frame = PERSONA_GUIDANCE.get(persona)
        if persona == "boss":
            rewritten += " I would appreciate your guidance on next steps."
        elif persona == "customer":
            rewritten += " Please let me know how this can be resolved."
        elif persona == "friend":
            rewritten += " I want to be honest without being hurtful."

        if context.strip():
            rewritten += " This reflects the broader context shared in the conversation."

        return self._post_process_rewrite(rewritten)

    def _post_process_rewrite(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        if cleaned and not cleaned.endswith((".", "!", "?")):
            cleaned += "."
        return cleaned

    def evaluate_transformation(self, original_text: str, rewritten_text: str) -> EvaluationResult:
        toxicity_before = self.classify_toxicity(original_text)
        toxicity_after = self.classify_toxicity(rewritten_text)
        return build_evaluation_result(
            original_text=original_text,
            rewritten_text=rewritten_text,
            toxicity_before=toxicity_before,
            toxicity_after=toxicity_after,
        )
