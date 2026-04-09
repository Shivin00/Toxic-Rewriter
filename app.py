from __future__ import annotations

import csv
import html
from io import StringIO
import re

import streamlit as st

from toxic_rewriter.nlp import (
    ReplacementRecord,
    RewriteAnalysis,
    ToxicRewriterService,
    ToxicityDimensionScore,
)
from toxic_rewriter.scoring import EvaluationResult, ToneProfile


st.set_page_config(
    page_title="Toxic Rewriter",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLES = {
    "Workplace feedback": "This report is a mess and your analysis is useless.",
    "Gaming chat": "You are so bad at this game, stop dragging the team down.",
    "Customer support": "Your service is terrible and nobody there knows what they are doing.",
    "Passive aggressive": "No offense, but you clearly have no idea what you're doing.",
    "Boundary setting": "You never listen, and I am done repeating myself.",
}

STYLE_HELP = {
    "formal professional": "Formal and workplace-ready with a respectful tone.",
    "casual friendly": "Softer, human, and conversational without sounding stiff.",
    "empathetic supportive": "Validates emotion and reduces accusation.",
    "assertive non-toxic": "Keeps boundaries and firmness while removing hostility.",
    "therapeutic": "Calm, reflective, and non-judgmental.",
    "diplomatic corporate": "Escalation-safe language for teams, managers, and clients.",
    "neutral factual": "Direct, concise, and minimally emotional.",
}

PERSONA_OPTIONS = ["boss", "friend", "customer", "coworker", "partner", "public audience"]
EMOTION_OPTIONS = ["calm", "collaborative", "constructive", "hopeful", "firm"]


@st.cache_resource(show_spinner=False)
def get_service() -> ToxicRewriterService:
    return ToxicRewriterService()


def init_session_state() -> None:
    if "tone_profile" not in st.session_state:
        st.session_state["tone_profile"] = {
            "preferred_style": "formal professional",
            "helpful_votes": 0,
            "too_soft_votes": 0,
            "too_strong_votes": 0,
        }


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 12% 10%, rgba(240, 151, 122, 0.20), transparent 28%),
                radial-gradient(circle at 88% 12%, rgba(102, 155, 188, 0.18), transparent 24%),
                radial-gradient(circle at 50% 100%, rgba(233, 196, 106, 0.10), transparent 28%),
                linear-gradient(180deg, #fff6f1 0%, #f8efe8 40%, #efe4da 100%);
            color: #2a3038;
            font-family: "Trebuchet MS", "Segoe UI", sans-serif;
        }
        .block-container {
            padding-top: 1.8rem;
            padding-bottom: 2.4rem;
            max-width: 1220px;
        }
        .hero {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at top right, rgba(255, 255, 255, 0.16), transparent 30%),
                linear-gradient(135deg, rgba(125, 58, 84, 0.98), rgba(207, 101, 70, 0.94) 52%, rgba(235, 154, 74, 0.92));
            border: 1px solid rgba(255, 255, 255, 0.24);
            border-radius: 34px;
            padding: 2.4rem 2.2rem 1.9rem 2.2rem;
            color: #f8f4ee;
            box-shadow: 0 30px 90px rgba(98, 59, 57, 0.20);
            margin-bottom: 1.4rem;
        }
        .hero::after {
            content: "";
            position: absolute;
            inset: auto -4rem -4rem auto;
            width: 240px;
            height: 240px;
            background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 66%);
            border-radius: 999px;
        }
        .hero-kicker {
            text-transform: uppercase;
            letter-spacing: 0.22em;
            font-size: 0.74rem;
            opacity: 0.82;
            margin-bottom: 0.7rem;
        }
        .hero-title {
            font-family: Georgia, "Times New Roman", serif;
            font-size: 3.4rem;
            font-weight: 700;
            line-height: 1.02;
            margin-bottom: 0.8rem;
            max-width: 700px;
        }
        .hero-copy {
            max-width: 820px;
            color: rgba(248, 244, 238, 0.9);
            font-size: 1.02rem;
            line-height: 1.75;
        }
        .hero-band {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 1rem;
        }
        .hero-pill {
            background: rgba(255, 246, 240, 0.14);
            border: 1px solid rgba(255,255,255,0.18);
            color: #fff3ee;
            border-radius: 999px;
            padding: 0.45rem 0.8rem;
            font-size: 0.84rem;
        }
        .glass-card {
            background: rgba(255, 250, 246, 0.88);
            border: 1px solid rgba(130, 103, 93, 0.10);
            border-radius: 24px;
            padding: 1rem 1rem 0.95rem 1rem;
            box-shadow: 0 18px 44px rgba(118, 92, 85, 0.08);
        }
        .mini-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.88), rgba(255,248,243,0.92));
            border-radius: 20px;
            border: 1px solid rgba(123, 111, 108, 0.10);
            padding: 0.85rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 10px 26px rgba(124, 104, 96, 0.06);
        }
        .mini-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #9a6d60;
            margin-bottom: 0.35rem;
        }
        .mini-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: #22323a;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.4rem;
        }
        .pill {
            background: rgba(204, 111, 83, 0.08);
            color: #8d4b3d;
            border: 1px solid rgba(204, 111, 83, 0.12);
            border-radius: 999px;
            padding: 0.42rem 0.75rem;
            font-size: 0.83rem;
        }
        .swap-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.8rem;
        }
        .swap-card {
            background: rgba(255, 251, 247, 0.96);
            border: 1px solid rgba(141, 123, 118, 0.14);
            border-radius: 20px;
            padding: 0.9rem;
            box-shadow: 0 10px 24px rgba(119, 94, 89, 0.06);
        }
        .swap-from {
            color: #a34141;
            font-weight: 600;
            margin-bottom: 0.45rem;
        }
        .swap-arrow {
            color: #9ca3af;
            margin-bottom: 0.45rem;
        }
        .swap-to {
            color: #17603c;
            font-weight: 600;
        }
        .styled-text {
            background: linear-gradient(180deg, rgba(255,255,255,0.72), rgba(255,249,245,0.88));
            border: 1px solid rgba(124, 113, 108, 0.10);
            border-radius: 22px;
            padding: 1.05rem;
            min-height: 180px;
            line-height: 1.75;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.5);
        }
        .highlight-toxic {
            background: rgba(187, 37, 37, 0.12);
            color: #8b1e1e;
            border-radius: 8px;
            padding: 0.05rem 0.28rem;
            font-weight: 600;
        }
        .highlight-polite {
            background: rgba(27, 94, 32, 0.12);
            color: #1d5b2d;
            border-radius: 8px;
            padding: 0.05rem 0.28rem;
            font-weight: 600;
        }
        .flow-step {
            background: rgba(255, 252, 247, 0.95);
            border-left: 4px solid #c66546;
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 8px 22px rgba(128, 107, 101, 0.05);
        }
        .flow-title {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: #7b6b57;
            margin-bottom: 0.35rem;
        }
        .flow-body {
            color: #23333d;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 247, 242, 0.92);
            border-radius: 999px;
            padding: 0.6rem 1rem;
            border: 1px solid rgba(146, 124, 117, 0.12);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(198,101,70,0.18), rgba(125,58,84,0.16));
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(255,248,243,0.95), rgba(248,237,229,0.98));
            border-right: 1px solid rgba(146, 124, 117, 0.10);
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 251, 247, 0.8);
            padding: 0.6rem 0.8rem;
            border-radius: 16px;
            border: 1px solid rgba(144, 122, 115, 0.10);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Context-Aware Tone Transformation</div>
            <div class="hero-title">Toxic Rewriter</div>
            <div class="hero-copy">
                Turn emotionally charged messages into language that still feels honest, human,
                and intentional. The app helps preserve what you mean while softening what harms.
            </div>
            <div class="hero-band">
                <div class="hero-pill">Emotion-aware rewriting</div>
                <div class="hero-pill">Intent preservation</div>
                <div class="hero-pill">Hidden toxicity detection</div>
                <div class="hero-pill">Safer communication</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def make_csv(rows: list[dict[str, object]]) -> str:
    output = StringIO()
    if not rows:
        return ""
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="mini-label">{html.escape(label)}</div>
            <div class="mini-value">{html.escape(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tone_table(before: ToneProfile, after: ToneProfile) -> list[dict[str, object]]:
    return [
        {"Signal": "Courtesy phrases", "Before": before.courtesy_hits, "After": after.courtesy_hits},
        {"Signal": "Harsh words", "Before": before.harsh_hits, "After": after.harsh_hits},
        {"Signal": "Aggressive patterns", "Before": before.aggressive_hits, "After": after.aggressive_hits},
        {"Signal": "Softeners", "Before": before.softener_hits, "After": after.softener_hits},
        {"Signal": "Exclamation marks", "Before": before.exclamation_count, "After": after.exclamation_count},
        {"Signal": "Uppercase ratio", "Before": round(before.uppercase_ratio, 3), "After": round(after.uppercase_ratio, 3)},
    ]


def highlight_fragment(rendered: str, phrase: str, css_class: str) -> str:
    if not phrase.strip():
        return rendered
    pattern = re.compile(re.escape(html.escape(phrase)), re.IGNORECASE)
    return pattern.sub(
        lambda match: f'<span class="{css_class}">{match.group(0)}</span>',
        rendered,
        count=1,
    )


def highlight_text(text: str, replacements: list[ReplacementRecord], target: str) -> str:
    rendered = html.escape(text)
    if target == "original":
        items = sorted(replacements, key=lambda item: len(item.source), reverse=True)
        for item in items:
            rendered = highlight_fragment(rendered, item.source, "highlight-toxic")
    else:
        items = sorted(replacements, key=lambda item: len(item.replacement), reverse=True)
        for item in items:
            rendered = highlight_fragment(rendered, item.replacement, "highlight-polite")
    return f'<div class="styled-text">{rendered}</div>'


def dimension_rows(items: list[ToxicityDimensionScore]) -> list[dict[str, object]]:
    return [
        {
            "Dimension": item.name,
            "Score": round(item.score, 3),
            "Evidence": ", ".join(item.evidence) if item.evidence else "-",
        }
        for item in items
    ]


def render_dimension_table(
    title: str,
    before_items: list[ToxicityDimensionScore],
    after_items: list[ToxicityDimensionScore] | None = None,
) -> None:
    st.subheader(title)
    rows: list[dict[str, object]] = []
    after_lookup = {item.name: item for item in (after_items or [])}
    for before in before_items:
        after = after_lookup.get(before.name)
        row = {
            "Dimension": before.name,
            "Before": round(before.score, 3),
            "Evidence": ", ".join(before.evidence) if before.evidence else "-",
        }
        if after is not None:
            row["After"] = round(after.score, 3)
        rows.append(row)
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_replacement_cards(replacements: list[ReplacementRecord]) -> None:
    if not replacements:
        st.info("No phrase-level substitutions were isolated, but the sentence structure still shifted.")
        return

    cards = []
    for item in replacements:
        cards.append(
            f"""
            <div class="swap-card">
                <div class="swap-from">{html.escape(item.source)}</div>
                <div class="swap-arrow">rewritten as</div>
                <div class="swap-to">{html.escape(item.replacement)}</div>
            </div>
            """
        )
    st.markdown(f'<div class="swap-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_flow_steps(flow_steps: list[dict[str, str]]) -> None:
    blocks = []
    for step in flow_steps:
        blocks.append(
            f"""
            <div class="flow-step">
                <div class="flow-title">{html.escape(step["stage"])}</div>
                <div class="flow-body">{html.escape(step["summary"])}</div>
            </div>
            """
        )
    st.markdown("".join(blocks), unsafe_allow_html=True)


def render_metric_grid(result: EvaluationResult) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Toxicity Before", format_percent(result.toxicity_before))
        render_metric_card("Politeness Before", format_percent(result.politeness_before))
    with col2:
        render_metric_card("Toxicity After", format_percent(result.toxicity_after))
        render_metric_card("Politeness After", format_percent(result.politeness_after))
    with col3:
        render_metric_card("Reduction", format_percent(result.toxicity_delta))
        render_metric_card("Assertiveness After", format_percent(result.assertiveness_after))
    with col4:
        render_metric_card("Meaning Retention", format_percent(result.semantic_similarity))
        render_metric_card("Professionalism", format_percent(result.professionalism_after))

    st.progress(result.confidence_score, text=f"Rewrite quality confidence: {format_percent(result.confidence_score)}")


def render_quality_cards(result: EvaluationResult) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Empathy Score", format_percent(result.empathy_after))
    with col2:
        st.metric("Clarity Score", format_percent(result.clarity_after))
    with col3:
        st.metric("Assertiveness Shift", format_percent(result.assertiveness_after - result.assertiveness_before))


def render_feedback_panel(style: str) -> None:
    left, mid, right = st.columns(3)
    if left.button("Helpful Rewrite", use_container_width=True):
        st.session_state["tone_profile"]["helpful_votes"] += 1
        st.session_state["tone_profile"]["preferred_style"] = style
    if mid.button("Too Soft", use_container_width=True):
        st.session_state["tone_profile"]["too_soft_votes"] += 1
    if right.button("Too Strong", use_container_width=True):
        st.session_state["tone_profile"]["too_strong_votes"] += 1


def render_analysis(original_text: str, analysis: RewriteAnalysis) -> None:
    result = analysis.evaluation

    text_left, text_right = st.columns(2)
    with text_left:
        st.subheader("Original")
        st.markdown(highlight_text(original_text, analysis.replacements, "original"), unsafe_allow_html=True)
    with text_right:
        st.subheader("Rewritten")
        st.markdown(highlight_text(analysis.rewritten_text, analysis.replacements, "rewritten"), unsafe_allow_html=True)

    render_metric_grid(result)
    render_quality_cards(result)

    chart_rows = [
        {"Stage": "Before", "Toxicity": result.toxicity_before, "Politeness": result.politeness_before, "Assertiveness": result.assertiveness_before},
        {"Stage": "After", "Toxicity": result.toxicity_after, "Politeness": result.politeness_after, "Assertiveness": result.assertiveness_after},
    ]

    chart_col, notes_col = st.columns([1.05, 0.95])
    with chart_col:
        st.subheader("Communication Shift")
        st.bar_chart(chart_rows, x="Stage", y=["Toxicity", "Politeness", "Assertiveness"])
    with notes_col:
        st.subheader("Rewrite Notes")
        for note in analysis.notes:
            st.write(f"- {note}")
        st.caption(f"Selected style: {analysis.style.title()}")

    detail_tab, explain_tab, safety_tab = st.tabs(["Toxicity Map", "Explainability Flow", "Safety Layer"])

    with detail_tab:
        render_dimension_table("Multi-Dimension Toxicity", analysis.original_dimensions, analysis.rewritten_dimensions)
        st.subheader("Hidden Toxicity Signals")
        st.dataframe(dimension_rows(analysis.hidden_toxicity), use_container_width=True, hide_index=True)

    with explain_tab:
        st.subheader("Rewrite Pipeline")
        render_flow_steps(analysis.flow_steps)
        st.subheader("Sentence-Level Replacements")
        render_replacement_cards(analysis.replacements)
        st.subheader("Intent Preservation")
        st.write(f"Original intent: {analysis.intent.original_intent}")
        st.write(f"Goal: `{analysis.intent.goal}`")
        st.write(f"Target entity: `{analysis.intent.target_entity}`")
        st.write(f"Desired emotional direction: `{analysis.intent.emotion}`")

    with safety_tab:
        if analysis.safety.flagged:
            st.warning(f"Residual safety risk detected: {format_percent(analysis.safety.overall_risk)}")
        else:
            st.success(f"Safety layer risk estimate: {format_percent(analysis.safety.overall_risk)}")
        for reason in analysis.safety.reasons:
            st.write(f"- {reason}")

    st.subheader("Tone Profile")
    st.dataframe(tone_table(result.tone_before, result.tone_after), use_container_width=True, hide_index=True)

    render_feedback_panel(analysis.style)

    export_rows = [
        {
            "style": analysis.style,
            "goal": analysis.intent.goal,
            "target_entity": analysis.intent.target_entity,
            "original_text": original_text,
            "rewritten_text": analysis.rewritten_text,
            "toxicity_before": round(result.toxicity_before, 4),
            "toxicity_after": round(result.toxicity_after, 4),
            "politeness_before": round(result.politeness_before, 4),
            "politeness_after": round(result.politeness_after, 4),
            "assertiveness_after": round(result.assertiveness_after, 4),
            "empathy_after": round(result.empathy_after, 4),
            "clarity_after": round(result.clarity_after, 4),
            "professionalism_after": round(result.professionalism_after, 4),
            "safety_risk": round(analysis.safety.overall_risk, 4),
        }
    ]
    st.download_button(
        "Download Result as CSV",
        data=make_csv(export_rows),
        file_name="toxic_rewriter_result.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_single_text_tab(service: ToxicRewriterService) -> None:
    left, right = st.columns([1.45, 0.9])

    with left:
        sample_name = st.selectbox("Choose a sample or write your own", ["Custom"] + list(EXAMPLES.keys()))
        default_text = "" if sample_name == "Custom" else EXAMPLES[sample_name]
        text = st.text_area(
            "Input text",
            value=default_text,
            height=180,
            placeholder="Paste harsh, passive-aggressive, or toxic text here...",
            key="single_input",
        )
        context = st.text_area(
            "Conversation context",
            height=90,
            placeholder="Optional context: what happened before this message?",
        )
        live_preview = st.checkbox("Live preview", value=True)
        run_clicked = st.button("Analyze and Rewrite", type="primary", use_container_width=True)

    with right:
        style = st.selectbox(
            "Politeness mode",
            options=list(STYLE_HELP.keys()),
            index=list(STYLE_HELP.keys()).index(st.session_state["tone_profile"]["preferred_style"]),
        )
        persona = st.selectbox("Persona-sensitive target", PERSONA_OPTIONS, index=3)
        emotion_target = st.selectbox("Emotional rewrite goal", EMOTION_OPTIONS, index=0)
        strength = st.slider("Firmness / strength preservation", min_value=0.0, max_value=1.0, value=0.55, step=0.05)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Mode Guide")
        st.write(STYLE_HELP[style])
        st.markdown(
            """
            <div class="pill-row">
                <div class="pill">Multi-axis toxicity</div>
                <div class="pill">Context-aware rewrite</div>
                <div class="pill">Intent preservation</div>
                <div class="pill">Safety layer</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    should_run = (live_preview and text.strip()) or run_clicked
    if should_run:
        if not text.strip():
            st.warning("Please enter some text first.")
            return
        with st.spinner("Running context-aware toxicity analysis and rewrite..."):
            analysis = service.analyze_rewrite(
                original_text=text,
                style=style,
                context=context,
                persona=persona,
                emotion_target=emotion_target,
                strength=strength,
            )
        render_analysis(text, analysis)


def render_batch_tab(service: ToxicRewriterService) -> None:
    st.write("Run a small experiment with one sentence per line.")
    batch_style = st.selectbox("Batch mode style", options=list(STYLE_HELP.keys()), index=0)
    batch_persona = st.selectbox("Batch persona", PERSONA_OPTIONS, index=3)
    batch_context = st.text_input("Batch context", value="Team discussion after a frustrating incident")
    batch_input = st.text_area(
        "Batch input",
        value="\n".join(EXAMPLES.values()),
        height=220,
        key="batch_input",
    )

    if st.button("Run Batch Rewrite", use_container_width=True):
        texts = [line.strip() for line in batch_input.splitlines() if line.strip()]
        if not texts:
            st.warning("Please enter at least one line of text.")
            return

        with st.spinner("Processing batch rewrite experiment..."):
            rows = service.batch_rewrite(
                texts=texts,
                style=batch_style,
                context=batch_context,
                persona=batch_persona,
                emotion_target="calm",
                strength=0.55,
            )

        st.subheader("Batch Results")
        st.dataframe(rows, use_container_width=True, hide_index=True)

        avg_toxicity_drop = sum(row["toxicity_before"] - row["toxicity_after"] for row in rows) / len(rows)
        avg_empathy = sum(row["empathy_after"] for row in rows) / len(rows)
        avg_professionalism = sum(row["professionalism_after"] for row in rows) / len(rows)

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Toxicity Reduction", format_percent(avg_toxicity_drop))
        col2.metric("Average Empathy", format_percent(avg_empathy))
        col3.metric("Average Professionalism", format_percent(avg_professionalism))

        st.download_button(
            "Download Batch CSV",
            data=make_csv(rows),
            file_name="toxic_rewriter_batch.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_project_tab(service: ToxicRewriterService) -> None:
    st.subheader("Current System Capabilities")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            **Analysis Engine**
            - Multi-dimension toxicity scoring
            - Hidden-toxicity detection for passive aggression and manipulative politeness
            - Intent inference for complaint, disagreement, and boundary-setting
            - Persona-aware and context-aware rewriting
            - Strength slider to preserve firmness
            """
        )
    with right:
        st.markdown(
            """
            **Communication Quality**
            - Toxicity and politeness before/after
            - Assertiveness, empathy, clarity, and professionalism scores
            - Explainability flow from detection to rewrite
            - Heuristic safety layer for residual abuse risk
            - Session-local feedback profile without persistent storage
            """
        )

    st.subheader("Runtime Status")
    st.write(f"Toxicity backend: `{service.classification_backend}`")
    st.write(f"Rewrite backend: `{service.rewrite_backend}`")

    st.subheader("Privacy-by-Design Notes")
    st.write("This demo keeps feedback only in the current Streamlit session and does not persist user text to disk.")
    st.write("A future enterprise path could add encrypted storage, on-device inference, and stricter compliance controls.")


def main() -> None:
    init_session_state()
    inject_styles()
    service = get_service()

    render_hero()

    with st.sidebar:
        st.subheader("Session Tone Profile")
        st.write(f"Preferred style: `{st.session_state['tone_profile']['preferred_style']}`")
        st.write(f"Helpful rewrites: `{st.session_state['tone_profile']['helpful_votes']}`")
        st.write(f"Too soft votes: `{st.session_state['tone_profile']['too_soft_votes']}`")
        st.write(f"Too strong votes: `{st.session_state['tone_profile']['too_strong_votes']}`")
        st.info("Feedback is kept only in the current session to support a privacy-first demo.")

    studio_tab, batch_tab, project_tab = st.tabs(["Rewrite Studio", "Batch Lab", "Project Notes"])
    with studio_tab:
        render_single_text_tab(service)
    with batch_tab:
        render_batch_tab(service)
    with project_tab:
        render_project_tab(service)


if __name__ == "__main__":
    main()
