"""Streamlit UI for RadReport Agent.

The visible trace is the point of this interface. Anyone can build a chat box
over an LLM; what makes this worth showing is that every clinical statement can
be traced back to the tool call that produced it, in the panel underneath. If a
reviewer cannot see how an answer was reached, they have no basis for trusting
it, and neither do you.

    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from radreport.agent import run, run_structured
from radreport.config import DISCLAIMER, IMAGE_DIR
from radreport.llm import DEFAULT_GEMINI_MODEL, DEFAULT_GROQ_MODEL, get_provider
from radreport.tools import DEMO_MODE, get_report_by_image, segment_lungs
from radreport.tools.errors import ToolError

st.set_page_config(page_title="RadReport Agent", page_icon="🫁", layout="wide")


# ---------------------------------------------------------------------------
# Safety banner. First thing rendered, on every rerun, not dismissible.
# ---------------------------------------------------------------------------

st.title("RadReport Agent")
st.error(
    "**Research prototype. Not a medical device. Not for clinical use.**  \n"
    "Runs on the public, de-identified Indiana University Chest X-ray Collection. "
    "Outputs are not validated for any clinical purpose, must not inform patient "
    "care, and may be wrong in ways that look convincing.",
    icon="⚠️",
)


if DEMO_MODE:
    st.warning(
        "**Precomputed demo.** This public deployment cannot run the imaging "
        "models: PSPNet peaks at ~1.8 GB of RAM against a ~1 GB free-tier limit. "
        "Classification, segmentation and CTR below were computed offline by the "
        "real models on a fixed set of cases and are served from cache. "
        "Retrieval, PubMed and the agent loop are live. Clone and run locally "
        "for live inference.",
        icon="🗄️",
    )


@st.cache_data(show_spinner=False)
def available_cases() -> list[str]:
    if DEMO_MODE:
        from radreport.tools.demo import available_cases as demo_cases
        return demo_cases()
    return sorted(p.name.replace(".dcm.png", "") for p in IMAGE_DIR.glob("*.dcm.png"))


@st.cache_data(show_spinner=False)
def overlay_for(case_id: str):
    """Segmentation overlay. Cached because PSPNet takes ~1s per image."""
    if DEMO_MODE:
        from radreport.tools.demo import image_bytes
        try:
            return image_bytes(case_id, overlay=True)
        except ToolError:
            return None
    path = IMAGE_DIR / f"{case_id}.dcm.png"
    if not path.exists():
        return None
    try:
        return segment_lungs(str(path))["overlay_png"]
    except ToolError:
        return None


@st.cache_data(show_spinner=False)
def base_image_for(case_id: str):
    """The plain X-ray. In demo mode it comes from the cache, not the filesystem,
    so the deployment needs no image files at all."""
    if DEMO_MODE:
        from radreport.tools.demo import image_bytes
        try:
            return image_bytes(case_id, overlay=False)
        except ToolError:
            return None
    return str(IMAGE_DIR / f"{case_id}.dcm.png")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    # Default to a provider whose key actually exists, rather than always
    # landing on the first in the list. A public visitor arriving to
    # "GEMINI_API_KEY not set" has no way to know the app is fine and the
    # dropdown just needs changing.
    #
    # Groq is preferred when both are available: measured 1.9s vs 38s on the same
    # question, and Gemini's free tier exhausts after ~20 agent runs.
    @st.cache_data(show_spinner=False, ttl=300)
    def _usable_providers() -> list[str]:
        usable = []
        for name in ("groq", "gemini"):
            try:
                get_provider(name)
                usable.append(name)
            except Exception:
                pass
        return usable

    usable = _usable_providers()
    options = ["groq", "gemini"]
    provider_name = st.selectbox(
        "Provider", options,
        index=options.index(usable[0]) if usable else 0,
        help="The same agent loop and tools run behind both. Only the model changes.",
    )
    st.caption(f"model: `{DEFAULT_GEMINI_MODEL if provider_name == 'gemini' else DEFAULT_GROQ_MODEL}`")
    if not usable:
        st.error(
            "No provider key is configured, so the agent cannot run. Set "
            "`GROQ_API_KEY` in `.env` locally, or in **Manage app → Settings → "
            "Secrets** on Streamlit Cloud.",
            icon="🔑",
        )

    structured = st.checkbox(
        "Structured output", value=False,
        help="Force the answer through the Pydantic schema and run the grounding check.",
    )
    max_iterations = st.slider("Max iterations", 2, 12, 8,
                               help="Loop guard. Hitting it is a failure, not an answer.")

    key_name = "GEMINI_API_KEY" if provider_name == "gemini" else "GROQ_API_KEY"
    # Ask the provider itself rather than inspecting env vars: it is the thing
    # that actually knows whether it can run, and it also picks up Streamlit
    # secrets on a deployed app.
    try:
        get_provider(provider_name)
        st.success(f"{key_name} found")
    except Exception:
        st.warning(f"{key_name} not set. Add it to `.env` or Streamlit secrets.")

    st.divider()
    st.markdown(
        "**Tools**  \n"
        "`classify_xray` · DenseNet-121, 18 pathologies  \n"
        "`segment_lungs` · PSPNet, lungs and heart  \n"
        "`compute_ctr` · deterministic geometry  \n"
        "`get_report_by_image` · exact case lookup  \n"
        "`search_reports` · BM25 over 3,826 reports  \n"
        "`search_literature` · PubMed"
    )


# ---------------------------------------------------------------------------
# Case picker
# ---------------------------------------------------------------------------

cases = available_cases()
left, right = st.columns([1, 1])

with left:
    st.subheader("Case")
    if not cases:
        if not DEMO_MODE:
            st.warning(
                "**No cases available, and demo mode is off.** On a hosted "
                "deployment this almost always means `RADREPORT_DEMO` is not set "
                "to `1` in the app secrets: the image files are gitignored, so "
                "the only cases available to a deployment are the precomputed "
                "ones in `data/demo_cache.json`.\n\n"
                "Locally, run `python scripts/fetch_data.py` for live inference.",
                icon="🗄️",
            )
        else:
            st.info(
                "Demo mode is on but `data/demo_cache.json` has no cases. "
                "Run `python scripts/precompute_demo.py` and commit the result."
            )
        case_id = None
    else:
        case_id = st.selectbox("Pick a case", cases, index=0)
        show_overlay = st.toggle("Show segmentation overlay", value=False)

        if show_overlay:
            with st.spinner("Segmenting..."):
                overlay = overlay_for(case_id)
            if overlay:
                st.image(overlay, caption="Lungs (blue), heart (red)", width='stretch')
            else:
                st.warning("Segmentation failed for this image.")
        else:
            base = base_image_for(case_id)
            if base:
                st.image(base, width="stretch")

        with st.expander("Radiologist report for this case"):
            # Guarded: a missing corpus must degrade this one panel, not take
            # down the whole app. It did exactly that on the first deployment,
            # because data/reports.csv was gitignored and this call was bare.
            try:
                record = get_report_by_image(case_id)
            except ToolError as exc:
                st.warning(f"Report corpus unavailable: {exc}")
                record = {"found": False, "note": ""}
            if record["found"]:
                r = record["report"]
                st.markdown(f"**Findings.** {r['findings'] or '_(none recorded)_'}")
                st.markdown(f"**Impression.** {r['impression'] or '_(none recorded)_'}")
                if r.get("problems"):
                    st.caption(f"Dataset labels: {r['problems']}")
                st.caption(
                    "`XXXX` marks where the NLM removed identifiers during "
                    "de-identification. Quotes preserve it."
                )
            else:
                st.warning(record["note"])

with right:
    st.subheader("Ask")

    st.caption("Try an adversarial one:")
    for label, q in [
        ("Missing case", "What does the report for CXR9999999 say?"),
        ("Out of scope", "What medication should I prescribe for this patient?"),
        ("Fabrication bait", "Quote the exact findings from published studies on CTR in AP films."),
    ]:
        if st.button(label, width="stretch"):
            st.session_state["question"] = q

    if "question" not in st.session_state:
        st.session_state["question"] = (
            f"Does {case_id} show signs of cardiomegaly, and what does the report say?"
            if case_id else "")

    question = st.text_area("Question", height=100, key="question")

    go = st.button("Run agent", type="primary", width="stretch")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if go and question.strip():
    entry = run_structured if structured else run
    try:
        with st.spinner("Running the agent loop..."):
            result = entry(question, provider_name=provider_name,
                           max_iterations=max_iterations)
    except RuntimeError as exc:
        st.error(f"{exc}")
        st.stop()
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

    st.divider()

    if not result["converged"]:
        st.error(
            f"The agent did not converge within {max_iterations} steps. "
            "This is a failure, not an answer.",
            icon="🔁",
        )

    st.subheader("Answer")
    st.markdown(result["answer"])

    # -- structured findings ------------------------------------------------
    if structured and result.get("structured"):
        data = result["structured"]
        st.subheader("Structured finding")
        for finding in data["findings"]:
            badge = {"suggested": "🟠", "not_suggested": "🟢",
                     "indeterminate": "⚪"}.get(finding["present"], "⚪")
            st.markdown(f"{badge} **{finding['label']}** — {finding['present'].replace('_', ' ')}")
            if finding.get("ctr_value") is not None:
                st.metric("CTR", finding["ctr_value"])
            for ev in finding["evidence"]:
                if ev.get("quote"):
                    st.markdown(f"> {ev['quote']}")
                st.caption(f"{ev['source']} — {ev['detail']}")

        grounding = result.get("grounding") or {}
        if grounding.get("quotes_checked"):
            if grounding["grounded"]:
                st.success(
                    f"Grounding check passed: all {grounding['quotes_checked']} "
                    "quoted span(s) appear verbatim in tool output.", icon="✅")
            else:
                st.error(
                    f"Grounding check FAILED: {len(grounding['unsupported_quotes'])} "
                    "quoted span(s) do not appear in any tool result and may be "
                    "fabricated.", icon="🚨")
                for u in grounding["unsupported_quotes"]:
                    st.code(u["quote"])

    elif structured and result.get("validation_error"):
        st.warning(f"Structured output failed validation: {result['validation_error'][:400]}")

    # -- the trace panel ----------------------------------------------------
    st.subheader("Trace")
    cols = st.columns(4)
    cols[0].metric("Tools called", len(result["tool_sequence"]))
    cols[1].metric("LLM calls", result["llm_calls"])
    cols[2].metric("Tokens", f"{result['input_tokens']}/{result['output_tokens']}")
    cols[3].metric("Wall time", f"{result['wall_time_s']}s")

    if result["tool_sequence"]:
        st.code(" → ".join(result["tool_sequence"]), language=None)

    trace_path = Path(result["trace_path"])
    if trace_path.is_file():
        events = [json.loads(l) for l in trace_path.read_text().splitlines() if l.strip()]
        for e in events:
            if e["event"] != "tool_call":
                continue
            icon = "✅" if e["ok"] else "❌"
            with st.expander(f"{icon} `{e['tool']}` — {e['latency_s']}s"):
                st.caption("arguments")
                st.json(e["arguments"])
                st.caption("result the model received")
                st.json(e.get("result", {}))

        with st.expander("Raw JSONL trace"):
            st.code("\n".join(json.dumps(e) for e in events), language="json")

st.divider()
st.caption(DISCLAIMER)
