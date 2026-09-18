"""
app.py — Streamlit app for Prism: The Glass Box AI Agent.

A live, interactive demo where judges can:
  1. Chat with the agent in real-time
  2. Watch the span tree, flamegraph, and metrics update live
  3. Trigger tool failures and see recovery
  4. Explore context engineering visually

Run: streamlit run app.py
"""

import streamlit as st
import sys
import os
import json
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prism.trace import Tracer, SpanKind
from prism.llm import create_llm, DeterministicLLM
from prism.tools import ToolRegistry
from prism.context import ContextEngine, ContextBudget
from prism.agent import Agent

# ─────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔮 Prism — Glass Box AI Agent",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Custom CSS — Premium dark theme
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global overrides */
.stApp {
    background: linear-gradient(180deg, #0a0a0f 0%, #0d0d1a 50%, #0a0a0f 100%);
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 15, 25, 0.95);
    border-right: 1px solid rgba(139, 92, 246, 0.15);
}

/* Cards */
div[data-testid="stMetric"] {
    background: rgba(20, 20, 35, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 16px;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

div[data-testid="stMetric"]:hover {
    border-color: rgba(139, 92, 246, 0.3);
    box-shadow: 0 0 30px rgba(139, 92, 246, 0.1);
}

div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    background: linear-gradient(135deg, #8b5cf6, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #a1a1aa;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: rgba(20, 20, 35, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    backdrop-filter: blur(10px);
}

div[data-testid="stExpander"]:hover {
    border-color: rgba(139, 92, 246, 0.2);
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    letter-spacing: 0.02em;
}

/* Chat input */
div[data-testid="stChatInput"] textarea {
    font-family: 'Inter', sans-serif;
    background: rgba(20, 20, 35, 0.8) !important;
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    border-radius: 12px !important;
}

/* Chat messages */
div[data-testid="stChatMessage"] {
    background: rgba(20, 20, 35, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 14px;
    padding: 12px;
    backdrop-filter: blur(5px);
}

/* Code blocks */
code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
}

/* Hero title */
.hero-title {
    text-align: center;
    padding: 1rem 0 0.5rem 0;
}

.hero-title h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 50%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}

.hero-title p {
    color: #71717a;
    font-size: 0.95rem;
    font-weight: 300;
}

/* Span tree styling */
.span-tree {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.9;
    padding: 0.5rem;
}

.span-node {
    padding: 3px 6px;
    border-radius: 6px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: background 0.2s;
}

.span-node:hover {
    background: rgba(255, 255, 255, 0.04);
}

.span-kind {
    font-size: 0.6rem;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.kind-AGENT { background: rgba(139,92,246,0.2); color: #a78bfa; }
.kind-LLM { background: rgba(6,182,212,0.2); color: #22d3ee; }
.kind-TOOL { background: rgba(245,158,11,0.2); color: #fbbf24; }
.kind-CONTEXT { background: rgba(16,185,129,0.2); color: #34d399; }
.kind-SYSTEM { background: rgba(161,161,170,0.15); color: #a1a1aa; }

.span-dur { color: #52525b; font-size: 0.7rem; margin-left: 4px; }
.span-tok { color: #22d3ee; font-size: 0.7rem; }
.span-err { color: #f87171; font-size: 0.7rem; font-weight: 600; }

.error-card {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: 12px;
    padding: 14px;
    margin: 8px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}

.success-card {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 12px;
    padding: 14px;
    margin: 8px 0;
}

/* Flamebar */
.flame-bar {
    height: 30px;
    border-radius: 6px;
    margin: 3px 0;
    display: flex;
    align-items: center;
    padding: 0 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.9);
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    transition: filter 0.2s, transform 0.2s;
}

.flame-bar:hover {
    filter: brightness(1.25);
    transform: scaleY(1.08);
}

/* Context bar */
.ctx-bar-container {
    display: flex;
    height: 26px;
    border-radius: 6px;
    overflow: hidden;
    background: rgba(255,255,255,0.02);
    margin: 4px 0;
}

.ctx-seg {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.55rem;
    font-family: 'JetBrains Mono', monospace;
    color: rgba(255,255,255,0.7);
    overflow: hidden;
    min-width: 2px;
}

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    margin: 1.5rem 0;
}

/* Pulse animation for live indicator */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
.live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    animation: pulse 2s ease-in-out infinite;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Session State Initialization
# ─────────────────────────────────────────────────────────
if "tracer" not in st.session_state:
    st.session_state.tracer = Tracer(artifacts_dir="artifacts")
    st.session_state.provider = "deterministic"
    st.session_state.model_name = "deterministic"
    st.session_state.api_key = ""
    st.session_state.llm = create_llm("deterministic")
    st.session_state.tools = ToolRegistry(st.session_state.tracer)
    st.session_state.context = ContextEngine(tracer=st.session_state.tracer)
    st.session_state.agent = Agent(
        st.session_state.llm,
        st.session_state.tools,
        st.session_state.context,
        st.session_state.tracer,
    )
    st.session_state.messages = []
    st.session_state.responses = []
    st.session_state.fail_mode = False
    st.session_state.stress_mode = False
    st.session_state.agent_mode = "🟢 Normal Mode (All Tools Active)"
    st.session_state.pending_query = None


# ─────────────────────────────────────────────────────────
# Helper: Render span tree as HTML
# ─────────────────────────────────────────────────────────
def render_span_html(span, depth=0):
    """Render a span and its children as styled HTML."""
    icon = {"AGENT": "🤖", "LLM": "🧠", "TOOL": "🔧", "CONTEXT": "📦", "RETRIEVE": "🔍", "SYSTEM": "⚙️"}.get(span.kind.value, "•")
    dur = f"{span.duration_ms:.0f}ms"
    tok = ""
    if span.input_tokens or span.output_tokens:
        tok = f'<span class="span-tok">[{span.input_tokens}→{span.output_tokens}]</span>'
    err = ""
    if span.error:
        err = f'<span class="span-err">✗ {span.error[:50]}</span>'

    indent = "│&nbsp;&nbsp;" * depth
    html = f'{indent}<span class="span-node">{icon} <span class="span-kind kind-{span.kind.value}">{span.kind.value}</span> <strong>{span.name}</strong> {tok} {err} <span class="span-dur">{dur}</span></span><br/>'

    for child in span.children:
        html += render_span_html(child, depth + 1)
    return html


def render_flamegraph_html(spans):
    """Render a flamegraph as styled HTML bars."""
    timed = [s for s in spans if s.duration_ms > 0]
    if not timed:
        return "<p style='color:#52525b'>No timed spans yet.</p>"

    max_dur = max(s.duration_ms for s in timed)
    colors = {
        "AGENT": "linear-gradient(90deg, #7c3aed, #6d28d9)",
        "LLM": "linear-gradient(90deg, #0891b2, #0e7490)",
        "TOOL": "linear-gradient(90deg, #d97706, #b45309)",
        "CONTEXT": "linear-gradient(90deg, #059669, #047857)",
        "RETRIEVE": "linear-gradient(90deg, #2563eb, #1d4ed8)",
        "SYSTEM": "linear-gradient(90deg, #4b5563, #374151)",
    }
    html = ""
    for s in timed:
        pct = max(8, (s.duration_ms / max_dur) * 100)
        bg = colors.get(s.kind.value, colors["SYSTEM"])
        html += f'<div class="flame-bar" style="width:{pct}%;background:{bg}">{s.name} — {s.duration_ms:.0f}ms</div>'
    return html


def render_context_bars_html(history, budget_total):
    """Render context allocation bars."""
    if not history:
        return "<p style='color:#52525b'>No context snapshots yet.</p>"

    seg_colors = {
        "system_prompt": "#8b5cf6",
        "key_facts": "#ec4899",
        "rolling_summary": "#3b82f6",
        "recent_turns": "#06b6d4",
        "tool_results": "#f59e0b",
    }
    html = ""
    for entry in history:
        alloc = entry["allocation"]
        total_used = sum(alloc.values())
        html += f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
        html += f'<span style="width:50px;font-size:0.7rem;color:#52525b;font-family:JetBrains Mono;text-align:right">T{entry["turn"]}</span>'
        html += f'<div class="ctx-bar-container" style="flex:1">'
        for key, tokens in alloc.items():
            if tokens > 0:
                pct = (tokens / budget_total) * 100
                color = seg_colors.get(key, "#6b7280")
                label = key.split("_")[0] if pct > 8 else ""
                html += f'<div class="ctx-seg" style="width:{pct}%;background:{color}">{label}</div>'
        html += '</div>'
        html += f'<span style="width:40px;font-size:0.7rem;color:#52525b;font-family:JetBrains Mono">{entry["utilization_pct"]:.0f}%</span>'
        html += '</div>'

    # Legend
    html += '<div style="display:flex;flex-wrap:wrap;gap:12px;margin-top:10px;font-size:0.65rem;color:#71717a">'
    for key, color in seg_colors.items():
        html += f'<span>■ <span style="color:{color}">{key.replace("_", " ")}</span></span>'
    html += '</div>'
    return html


# ─────────────────────────────────────────────────────────
# Sidebar: Chat Interface
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title"><h1>🔮 Prism</h1><p>Glass Box AI Agent</p></div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── 1. Execution Mode Selector ──
    mode_options = [
        "🟢 Normal Mode (All Tools Active)",
        "⚠️ Tool Failure Mode (HTTP 503 Fault Injection)",
        "📦 Context Stress Mode (Tight Token Budget)",
    ]
    current_mode_idx = 0
    if st.session_state.fail_mode:
        current_mode_idx = 1
    elif st.session_state.stress_mode:
        current_mode_idx = 2

    chosen_mode = st.selectbox(
        "🎮 Agent Operating Mode",
        options=mode_options,
        index=current_mode_idx,
        help="Switch between standard agent execution, simulated tool network failures (to prove error recovery), and context budget stress testing."
    )

    if "Tool Failure" in chosen_mode:
        st.session_state.fail_mode = True
        st.session_state.stress_mode = False
        st.session_state.tools._fail_search = True
        st.session_state.tools._fail_url = True
        st.warning("⚠️ Tool failure mode active — search & URL tools will simulate HTTP 503 to demonstrate autonomous error recovery!", icon="⚠️")
    elif "Context Stress" in chosen_mode:
        st.session_state.fail_mode = False
        st.session_state.stress_mode = True
        st.session_state.tools._fail_search = False
        st.session_state.tools._fail_url = False
        st.session_state.context.budget = ContextBudget(
            system_prompt=150,
            key_facts=80,
            rolling_summary=150,
            recent_turns=300,
            tool_results=200,
            response_room=300,
        )
        st.info("📦 Context stress mode active — compact 1,180 token budget forces rapid rolling summary & FIFO trimming.", icon="📦")
    else:
        st.session_state.fail_mode = False
        st.session_state.stress_mode = False
        st.session_state.tools._fail_search = False
        st.session_state.tools._fail_url = False
        st.session_state.context.budget = ContextBudget()

    # ── 2. LLM Provider & Model Settings ──
    with st.expander("⚙️ LLM Provider & Model Settings", expanded=False):
        provider_display = {
            "deterministic": "🤖 Deterministic (Offline / Mock / Zero-Key)",
            "groq": "⚡ Groq (Free Tier · Ultra Fast Llama 3.3 70B)",
            "gemini": "💎 Google Gemini (gemini-3.6-flash)",
            "openrouter": "🌐 OpenRouter (Free Community Models)",
            "openai": "🧠 OpenAI GPT (gpt-4o-mini / gpt-4o)",
            "anthropic": "🎭 Anthropic Claude",
        }
        provider_keys = list(provider_display.keys())
        current_p = st.session_state.get("provider", "deterministic")
        p_index = provider_keys.index(current_p) if current_p in provider_keys else 0

        chosen_provider = st.selectbox(
            "AI Provider",
            options=provider_keys,
            index=p_index,
            format_func=lambda k: provider_display[k],
        )

        chosen_model = None
        user_api_key = None

        if chosen_provider == "groq":
            default_groq = [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "groq/compound",
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "Custom (enter below)",
            ]
            detected = st.session_state.get("groq_detected_models", [])
            groq_models = detected if detected else default_groq

            selected_option = st.selectbox("Groq Model", groq_models)
            if selected_option == "Custom (enter below)":
                chosen_model = st.text_input("Enter Groq Model ID", value="openai/gpt-oss-120b")
            else:
                chosen_model = selected_option

            env_key = os.environ.get("GROQ_API_KEY") or (st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None)
            if env_key:
                st.caption("✅ Key detected from environment/secrets")
            st.caption("🔑 Free key in 30 seconds (no credit card): [console.groq.com/keys](https://console.groq.com/keys)")
            user_api_key = st.text_input(
                "Groq API Key (starts with gsk_)",
                value=st.session_state.get("api_key_groq", "") or (env_key or ""),
                type="password",
                key="input_api_key_groq"
            )
            if user_api_key:
                st.session_state["api_key_groq"] = user_api_key
                if not user_api_key.startswith("gsk_"):
                    st.warning("⚠️ Warning: Groq API keys start with **gsk_**. It looks like you pasted a key from another provider (e.g. Gemini or OpenAI).")
                else:
                    if st.button("🔍 Test Key & List My Accessible Groq Models", key="btn_test_groq"):
                        with st.spinner("Connecting to Groq API..."):
                            try:
                                import requests
                                r = requests.get(
                                    "https://api.groq.com/openai/v1/models",
                                    headers={"Authorization": f"Bearer {user_api_key}"},
                                    timeout=6,
                                )
                                if r.status_code == 200:
                                    m_data = r.json().get("data", [])
                                    valid_ids = [m["id"] for m in m_data if not m.get("id", "").startswith("whisper")]
                                    if valid_ids:
                                        st.session_state["groq_detected_models"] = sorted(valid_ids)
                                        st.success(f"✅ Key Verified! {len(valid_ids)} active models loaded into dropdown.")
                                        st.rerun()
                                    else:
                                        st.info("Key verified, but no chat models returned.")
                                else:
                                    st.error(f"❌ Groq API rejected key ({r.status_code}): {r.text[:200]}")
                            except Exception as ex:
                                st.error(f"❌ Connection error: {ex}")
            else:
                st.info("💡 Enter your free Groq API key above, or switch to **Deterministic** mode below for instant zero-key testing.")

        elif chosen_provider == "openrouter":
            openrouter_models = [
                "meta-llama/llama-3.3-70b-instruct:free",
                "google/gemini-2.0-flash-exp:free",
                "qwen/qwen-2.5-72b-instruct:free",
                "deepseek/deepseek-r1:free",
                "mistralai/mistral-7b-instruct:free",
            ]
            chosen_model = st.selectbox("OpenRouter Model", openrouter_models)
            env_key = os.environ.get("OPENROUTER_API_KEY") or (st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, "secrets") else None)
            if env_key:
                st.caption("✅ Key detected from environment/secrets")
            st.caption("🔑 Free key at [openrouter.ai/keys](https://openrouter.ai/keys)")
            user_api_key = st.text_input(
                "OpenRouter API Key (starts with sk-or-)",
                value=st.session_state.get("api_key_openrouter", "") or (env_key or ""),
                type="password",
                key="input_api_key_openrouter"
            )
            if user_api_key:
                st.session_state["api_key_openrouter"] = user_api_key
                if not user_api_key.startswith("sk-or-"):
                    st.warning("⚠️ Warning: OpenRouter API keys usually start with **sk-or-**.")
            else:
                st.info("💡 Enter your OpenRouter key above, or switch to **Deterministic** mode for zero-key testing.")

        elif chosen_provider == "gemini":
            chosen_model = st.selectbox("Gemini Model", ["gemini-3.6-flash", "gemini-flash-latest", "gemini-3.8-flash", "gemini-pro-latest"])
            env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or (st.secrets.get("GEMINI_API_KEY") if hasattr(st, "secrets") else None)
            if env_key:
                st.caption("✅ Key detected from environment/secrets")
            st.caption("🔑 Free Google key: [aistudio.google.com](https://aistudio.google.com)")
            user_api_key = st.text_input(
                "Gemini API Key (starts with AIza / AQ.)",
                value=st.session_state.get("api_key_gemini", "") or (env_key or ""),
                type="password",
                key="input_api_key_gemini"
            )
            if user_api_key:
                st.session_state["api_key_gemini"] = user_api_key
                if not (user_api_key.startswith("AIza") or user_api_key.startswith("AQ.")):
                    st.warning("⚠️ Warning: Google Gemini keys typically start with **AIza** or **AQ.**.")
            else:
                st.info("💡 Enter your Gemini key above, or switch to **Deterministic** mode for zero-key testing.")

        elif chosen_provider == "openai":
            chosen_model = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
            env_key = os.environ.get("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
            if env_key:
                st.caption("✅ Key detected from environment/secrets")
            user_api_key = st.text_input(
                "OpenAI API Key (starts with sk-)",
                value=st.session_state.get("api_key_openai", "") or (env_key or ""),
                type="password",
                key="input_api_key_openai"
            )
            if user_api_key:
                st.session_state["api_key_openai"] = user_api_key

        elif chosen_provider == "anthropic":
            chosen_model = st.selectbox("Claude Model", ["claude-haiku-3.5", "claude-3-5-sonnet-20241022"])
            env_key = os.environ.get("ANTHROPIC_API_KEY") or (st.secrets.get("ANTHROPIC_API_KEY") if hasattr(st, "secrets") else None)
            if env_key:
                st.caption("✅ Key detected from environment/secrets")
            user_api_key = st.text_input(
                "Anthropic API Key (starts with sk-ant-)",
                value=st.session_state.get("api_key_anthropic", "") or (env_key or ""),
                type="password",
                key="input_api_key_anthropic"
            )
            if user_api_key:
                st.session_state["api_key_anthropic"] = user_api_key
        else:
            chosen_model = "deterministic"
            user_api_key = ""

        # Update LLM if changed
        current_active_p = st.session_state.get("provider")
        current_active_m = st.session_state.get("model_name")
        current_active_k = st.session_state.get("api_key")

        if (chosen_provider != current_active_p or
            chosen_model != current_active_m or
            user_api_key != current_active_k):
            try:
                # If an API provider is chosen but no key is available, fallback safely to Deterministic
                effective_provider = chosen_provider
                effective_model = chosen_model
                if chosen_provider != "deterministic" and not user_api_key:
                    effective_provider = "deterministic"
                    effective_model = "deterministic"

                new_llm = create_llm(provider=effective_provider, model=effective_model, api_key=user_api_key or None)
                st.session_state.llm = new_llm
                st.session_state.provider = chosen_provider
                st.session_state.model_name = chosen_model
                st.session_state.api_key = user_api_key
                st.session_state.agent = Agent(
                    st.session_state.llm,
                    st.session_state.tools,
                    st.session_state.context,
                    st.session_state.tracer,
                )
                st.success(f"Switched to {chosen_provider} ({chosen_model})")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to switch provider: {e}")

    # Active status badge
    st.markdown(
        f"<div style='font-size:0.75rem;padding:6px 10px;border-radius:8px;background:rgba(139,92,246,0.12);border:1px solid rgba(139,92,246,0.25);color:#c4b5fd;margin:6px 0 10px 0'>"
        f"🧠 Active Model: <b>{st.session_state.llm.model_name}</b>"
        f"</div>",
        unsafe_allow_html=True
    )

    # ── 3. Reset Session Button ──
    if st.button("🔄 Reset Session & Clear Traces", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    # ── 4. 1-Click Demo Queries ──
    with st.expander("⚡ 1-Click Demo Queries", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔍 Search Agent", use_container_width=True):
                st.session_state.pending_query = "Search for latest developments in AI agent observability."
                st.rerun()
            if st.button("💥 Failure Test", use_container_width=True):
                st.session_state.fail_mode = True
                st.session_state.tools._fail_search = True
                st.session_state.pending_query = "Search for quantum computing breakthroughs."
                st.rerun()
            if st.button("🌤️ Live Weather", use_container_width=True):
                st.session_state.pending_query = "What is the current weather in Tokyo?"
                st.rerun()
        with c2:
            if st.button("🧮 Math Query", use_container_width=True):
                st.session_state.pending_query = "Calculate (45 * 12) + (180 / 4)"
                st.rerun()
            if st.button("📝 Take Note", use_container_width=True):
                st.session_state.pending_query = "Take note: Epochesque 2.0 Track 1 submission ready."
                st.rerun()
            if st.button("📖 Wikipedia", use_container_width=True):
                st.session_state.pending_query = "Wikipedia summary of Alan Turing"
                st.rerun()

    st.markdown("---")
    st.markdown("**💬 Chat**")

    # Message history display
    chat_container = st.container(height=380)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🔮"):
                st.markdown(msg["content"])
                if "meta" in msg:
                    st.caption(msg["meta"])

    # Chat input & pending query execution
    incoming_query = None
    if st.session_state.get("pending_query"):
        incoming_query = st.session_state.pop("pending_query")

    typed_input = st.chat_input("Ask me anything...")
    if typed_input:
        incoming_query = typed_input

    if incoming_query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": incoming_query})

        try:
            # Run the agent
            with st.spinner("🔮 Agent planning and executing tools..."):
                response = st.session_state.agent.run(incoming_query)
            st.session_state.responses.append(response)

            # Build meta string
            tool_steps = [s for s in response.steps if s.action == "tool_call"]
            error_steps = [s for s in response.steps if s.action == "error_recovery"]
            meta_parts = [f"{response.total_steps} steps"]
            if tool_steps:
                meta_parts.append(f"tools: {', '.join(s.tool_name for s in tool_steps if s.tool_name)}")
            if error_steps:
                meta_parts.append(f"⚠️ {len(error_steps)} error(s) recovered")
            meta_parts.append(f"{response.tokens_used:,} tok")
            meta_parts.append(f"${response.cost_usd:.6f}")
            meta_parts.append(f"{response.duration_ms:.0f}ms")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response.content,
                "meta": " · ".join(meta_parts),
            })
        except Exception as e:
            provider_now = st.session_state.get("provider", "deterministic")
            hint = ""
            if provider_now == "gemini":
                hint = "\n\n*Tip: Free-tier Gemini keys have rate limits (e.g. 5-15 req/min) or queue delays. You can wait ~30s, try Groq / OpenRouter in the sidebar, or switch to `Deterministic` mode for instant offline execution.*"
            elif provider_now == "groq":
                hint = "\n\n*Tip: Groq keys start with `gsk_` (free at [console.groq.com/keys](https://console.groq.com/keys)). Make sure you haven't pasted a Gemini key into the Groq field. You can also switch to `Deterministic` mode for instant execution.*"
            elif provider_now == "openrouter":
                hint = "\n\n*Tip: OpenRouter free models end with `:free` and keys start with `sk-or-` from [openrouter.ai/keys](https://openrouter.ai/keys). You can also switch to `Deterministic` mode.*"
            else:
                hint = "\n\n*Tip: You can switch to `Deterministic` mode in the sidebar for unlimited, instant offline execution.*"

            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ **Execution Notice**: `{e}`{hint}",
            })
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;color:#52525b;text-align:center'>"
        "Built for Epochesque 2.0 — Track 1<br/>Team Imagine Losing"
        "</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────
# Main Area: Live Dashboard
# ─────────────────────────────────────────────────────────

# Hero
st.markdown(
    '<div class="hero-title">'
    '<h1>🔮 Prism Dashboard</h1>'
    '<p><span class="live-dot"></span>Live — Every Decision Traced</p>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Metrics Row ──
summary = st.session_state.tracer.summary()
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("🧠 LLM Calls", summary["llm_calls"])
m2.metric("🔧 Tool Calls", summary["tool_calls"])
m3.metric("📊 Tokens", f'{summary["total_tokens"]:,}')
m4.metric("💰 Cost", f'${summary["cost_usd"]:.6f}')
m5.metric("⏱️ Time", f'{summary["wall_ms"]:.0f}ms')
m6.metric("⚠️ Errors", summary["errors"])

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── Tabs ──
tab_tree, tab_flame, tab_tokens, tab_context, tab_tools, tab_failures, tab_replay = st.tabs([
    "🌳 Span Tree",
    "🔥 Flamegraph",
    "📊 Token Flow",
    "📦 Context Window",
    "🔧 Tool Usage",
    "⚠️ Failures",
    "💬 Replay",
])

# ── Tab: Span Tree ──
with tab_tree:
    spans = st.session_state.tracer.get_root_spans()
    if spans:
        tree_html = '<div class="span-tree">'
        for root in spans:
            tree_html += render_span_html(root)
        tree_html += '</div>'
        st.markdown(tree_html, unsafe_allow_html=True)
    else:
        st.info("No spans yet. Send a message in the sidebar to start tracing!")

# ── Tab: Flamegraph ──
with tab_flame:
    all_spans = st.session_state.tracer.get_spans()
    if all_spans:
        st.markdown(render_flamegraph_html(all_spans), unsafe_allow_html=True)
    else:
        st.info("No timed spans yet.")

# ── Tab: Token Flow ──
with tab_tokens:
    llm_spans = [s for s in st.session_state.tracer.get_spans() if s.kind == SpanKind.LLM]
    if llm_spans:
        import plotly.graph_objects as go

        names = [s.name for s in llm_spans]
        inputs = [s.input_tokens for s in llm_spans]
        outputs = [s.output_tokens for s in llm_spans]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=names, x=inputs, name="Input Tokens",
            orientation="h",
            marker_color="#3b82f6",
            text=inputs, textposition="inside",
        ))
        fig.add_trace(go.Bar(
            y=names, x=outputs, name="Output Tokens",
            orientation="h",
            marker_color="#8b5cf6",
            text=outputs, textposition="inside",
        ))
        fig.update_layout(
            barmode="stack",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="JetBrains Mono, monospace", color="#a1a1aa", size=11),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        font=dict(color="#a1a1aa")),
            margin=dict(l=10, r=10, t=40, b=10),
            height=max(200, len(names) * 40 + 80),
            xaxis=dict(title="Tokens", gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)", autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No LLM calls yet.")

# ── Tab: Context Window ──
with tab_context:
    history = st.session_state.context.get_allocation_history()
    budget_total = st.session_state.context.budget.total
    if history:
        st.markdown("**Context budget allocation across turns:**")
        st.markdown(render_context_bars_html(history, budget_total), unsafe_allow_html=True)

        # Status
        status = st.session_state.context.get_status()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Turns", status["total_turns"])
        c2.metric("Window Size", status["window_size"])
        c3.metric("Key Facts", status["key_facts"])
        c4.metric("Has Summary", "Yes" if status["has_summary"] else "No")

        if history:
            import plotly.graph_objects as go

            turns = [e["turn"] for e in history]
            sections = ["system_prompt", "key_facts", "rolling_summary", "recent_turns"]
            colors = {"system_prompt": "#8b5cf6", "key_facts": "#ec4899",
                      "rolling_summary": "#3b82f6", "recent_turns": "#06b6d4"}

            fig = go.Figure()
            for sec in sections:
                values = [e["allocation"].get(sec, 0) for e in history]
                fig.add_trace(go.Bar(
                    x=turns, y=values, name=sec.replace("_", " ").title(),
                    marker_color=colors.get(sec, "#6b7280"),
                ))
            fig.update_layout(
                barmode="stack",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#a1a1aa", size=11),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=40, b=10),
                height=300,
                xaxis=dict(title="Turn", gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="Tokens", gridcolor="rgba(255,255,255,0.04)"),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No context snapshots yet. Start chatting to see allocation!")

# ── Tab: Tool Usage ──
with tab_tools:
    with st.expander("🛠️ Available Tools Registry (8 Active Tools)", expanded=False):
        t_cols = st.columns(4)
        tool_icons = {
            "web_search": "🔍",
            "read_url": "🌐",
            "calculate": "🧮",
            "analyze_data": "📊",
            "take_note": "📝",
            "get_weather": "🌤️",
            "wikipedia_summary": "📖",
            "datetime_info": "🕒",
        }
        for idx, schema in enumerate(st.session_state.tools.schemas):
            t_name = schema["name"]
            t_desc = schema["description"]
            t_icon = tool_icons.get(t_name, "🔧")
            with t_cols[idx % 4]:
                st.markdown(f"**{t_icon} `{t_name}`**")
                st.caption(t_desc[:90] + "...")

    tool_spans = [s for s in st.session_state.tracer.get_spans() if s.kind == SpanKind.TOOL]
    if tool_spans:
        tool_stats = {}
        for ts in tool_spans:
            name = ts.name.replace("tool:", "")
            if name not in tool_stats:
                tool_stats[name] = {"calls": 0, "ok": 0, "err": 0, "total_ms": 0}
            tool_stats[name]["calls"] += 1
            if ts.error:
                tool_stats[name]["err"] += 1
            else:
                tool_stats[name]["ok"] += 1
            tool_stats[name]["total_ms"] += ts.duration_ms

        cols = st.columns(min(len(tool_stats), 4))
        for i, (name, stats) in enumerate(tool_stats.items()):
            with cols[i % len(cols)]:
                rate = round((stats["ok"] / stats["calls"]) * 100) if stats["calls"] > 0 else 0
                avg_ms = round(stats["total_ms"] / stats["calls"]) if stats["calls"] > 0 else 0
                color = "🟢" if rate >= 80 else "🟡" if rate >= 50 else "🔴"

                st.markdown(f"### 🔧 {name}")
                st.markdown(f"**{stats['calls']}** calls · {color} **{rate}%** success")
                st.markdown(f"✅ {stats['ok']} ok · ❌ {stats['err']} err · ⏱️ {avg_ms}ms avg")
                st.progress(rate / 100)
    else:
        st.info("No tool calls yet. Select a 1-Click Demo Query in the sidebar or ask a question!")

# ── Tab: Failures ──
with tab_failures:
    error_spans = [s for s in st.session_state.tracer.get_spans() if s.error]
    if error_spans:
        for es in error_spans:
            events_str = ""
            for ev in es.events:
                if ev.attrs:
                    events_str += f"\n  → {ev.name}: {json.dumps(ev.attrs, default=str)}"

            st.markdown(
                f'<div class="error-card">'
                f'<strong>⚠️ {es.name}</strong> <span style="color:#71717a">({es.kind.value})</span><br/>'
                f'<code>{es.error}</code>'
                f'{f"<pre>{events_str}</pre>" if events_str else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="success-card">✅ <strong>No errors recorded</strong> — all operations succeeded.</div>',
            unsafe_allow_html=True,
        )

# ── Tab: Conversation Replay ──
with tab_replay:
    turns = st.session_state.context._turns
    if turns:
        for turn in turns:
            icon = "👤" if turn.role == "user" else "🤖" if turn.role == "assistant" else "🔧"
            tool_badge = f" `[{turn.tool_name}]`" if turn.tool_name else ""
            with st.expander(f"{icon} **{turn.role.upper()}**{tool_badge} — Turn #{turn.turn_number} · ~{turn.tokens} tokens", expanded=False):
                st.markdown(turn.content[:500])
    else:
        st.info("No conversation yet. Start chatting!")

# ── Footer ──
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center;color:#3f3f46;font-size:0.75rem;padding:0.5rem'>"
    f"🔮 Prism · Run {summary['run_id']} · {summary['span_count']} spans · "
    f"Epochesque 2.0 Track 1 · Team Imagine Losing"
    f"</div>",
    unsafe_allow_html=True,
)
