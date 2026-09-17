# 🔮 Prism — The Glass Box AI Agent

> **An AI research assistant you can see inside.** Every decision, every tool call, every token spent — traced, visualized, and explainable.

<p align="center">
  <strong>Built for <a href="#">Epochesque 2.0</a> — Track 1: "The Glass Box Problem"</strong>
</p>

---

## 📖 Table of Contents

1. [What Problem Are We Solving?](#-what-problem-are-we-solving)
2. [What Did We Build?](#-what-did-we-build)
3. [How It Works (For Beginners)](#-how-it-works-for-beginners)
4. [Architecture Deep Dive](#-architecture-deep-dive)
5. [Quick Start](#-quick-start)
6. [Demo Walkthrough](#-demo-walkthrough)
7. [Project Structure](#-project-structure)
8. [The Eval Suite](#-the-eval-suite)
9. [Context Engineering Explained](#-context-engineering-explained)
10. [LLM Provider Support](#-llm-provider-support)
11. [How We Meet the Judging Criteria](#-how-we-meet-the-judging-criteria)
12. [Known Limitations](#-known-limitations)

---

## 🧩 What Problem Are We Solving?

### The Hackathon Challenge

Epochesque 2.0 is a hackathon with three tracks. We chose **Track 1: "The Glass Box Problem."**

The premise is simple but critical:

> *Your AI pipeline works today. Nobody knows why it will break tomorrow, and when it does, nobody can tell what went wrong.*

Think about it: when you use ChatGPT or any AI tool, you type a question and get an answer. But what happens *inside*? Which tools did it use? Why did it pick that tool over another? How much did that call cost? What information was it actually looking at when it decided what to say?

**You have no idea.** It's a black box.

This is a real-world problem. Companies deploying AI agents in production need to:
- **Debug failures** — when the AI gives a wrong answer, you need to trace back and find *why*
- **Control costs** — LLM API calls cost real money; you need to know where tokens are going
- **Stay accurate** — in long conversations, the AI can "forget" important context or get confused
- **Trust the output** — you need to know *which source* the AI used for each claim

### The Rules We Had to Follow

1. ✅ Produce a **trace/log for every run** — not just a final answer
2. ✅ Show at least one **real failure case** caught via tracing
3. ✅ Handle **15-20+ turn conversations** without breaking
4. ✅ Track **token usage and cost** per run
5. ✅ If using off-the-shelf tools, explain what they track and why

---

## 🏗️ What Did We Build?

**Prism** is a multi-tool AI research agent with a built-in observability (tracing) layer. It has three major parts:

### 1. The Agent 🤖
A hand-written AI assistant that can:
- **Search the web** for information
- **Read web pages** and extract content
- **Do math** calculations safely
- **Analyze data** (numbers, CSVs)
- **Take notes** during research

It decides which tool to use on its own — no hardcoded rules like "if the user says 'search', call the search function." The LLM reads the available tools and picks the best one.

### 2. The Tracing Layer 🔍
Every single operation is wrapped in a **Span** — a timed unit of work. Spans nest inside each other like Russian dolls:

```
🤖 agent_run (the whole query)
  📦 context_engineering (building what the AI sees)
  🤖 step_1 (first reasoning step)
    🤖 plan (ask the LLM what to do)
      🧠 llm:plan_and_act (the actual API call — 199 input tokens, 12 output tokens)
    🔧 tool:web_search (execute the tool — 643ms)
  🤖 step_2 (second reasoning step)
    🤖 plan
      🧠 llm:plan_and_act (306 input tokens, 170 output tokens)
    → response synthesized
```

Every span records: start time, end time, token counts, cost, errors, and arbitrary metadata. This means you can answer questions like:
- "How long did the web search take?" → Look at the `tool:web_search` span
- "How many tokens did planning use?" → Sum the `llm:plan_and_act` spans
- "Did any tool fail?" → Check for spans with `error` set

### 3. The Dashboard 📊
A stunning, interactive HTML dashboard that visualizes everything:
- **Span Tree** — collapsible view of every operation
- **Flamegraph** — where time was spent
- **Token Flow** — input vs output tokens per LLM call
- **Context Window** — how the AI's "memory budget" is allocated
- **Tool Usage** — which tools were called and their success rates
- **Failure Cases** — every error caught and how the agent recovered
- **Conversation Replay** — the full chat with "under the hood" details

---

## 🎓 How It Works (For Beginners)

If you're new to AI agents and LLMs, here's a plain-English explanation of the key concepts:

### What's an LLM?
A **Large Language Model** (like GPT-4, Gemini, or Claude) is an AI that takes text in and gives text out. You send it a prompt, it sends back a response. Every call costs tokens (roughly 1 token ≈ 4 characters of text).

### What's an AI Agent?
A regular chatbot just answers questions. An **agent** can also *do things* — search the web, run calculations, read files. It follows a loop:

```
1. PLAN   — "What should I do to answer this question?"
2. ACT    — Call a tool (search, calculate, etc.)
3. OBSERVE — "Did the tool work? Do I have enough info?"
4. REPEAT  — If not done, go back to step 1
5. RESPOND — Synthesize a final answer
```

### What's Observability?
**Observability** means being able to see what's happening inside a system. For AI agents, this means:
- Logging every LLM API call (what went in, what came out, how many tokens, how much it cost)
- Logging every tool call (what tool, what arguments, did it succeed, how long)
- Tracking the full decision chain (why did the agent pick *this* tool over *that* one?)

### What's Context Engineering?
LLMs have a **context window** — a limit on how much text they can "see" at once. In a long conversation (20+ turns), you can't send the entire history because it would exceed the limit. **Context engineering** is the art of deciding what the AI should see:

- **Recent messages** — the last few turns (most important for continuing the conversation)
- **Summary of older messages** — compressed version of earlier context
- **Key facts** — important user preferences that should never be lost

Prism manages all of this with a token budget system.

### What's a Span / Trace?
Borrowed from software engineering, a **trace** is a record of everything that happened during one request. A **span** is one unit of work within that trace. Spans can be nested (a span can contain child spans), forming a tree that shows exactly how work was broken down.

---

## 🏛️ Architecture Deep Dive

Here's how data flows through Prism:

```
                    ┌──────────────┐
  User Query ──────►│  Agent Loop  │
                    │  (agent.py)  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌────────────┐ ┌───────────┐ ┌──────────┐
     │  Context   │ │    LLM    │ │   Tools  │
     │  Engine    │ │ Interface │ │ Registry │
     │(context.py)│ │ (llm.py)  │ │(tools.py)│
     └────────────┘ └───────────┘ └──────────┘
           │              │             │
           │         ┌────┴────┐        │
           │         │ Gemini  │        ├── web_search
           │         │ OpenAI  │        ├── read_url
           │         │ Claude  │        ├── calculate
           │         │  Stub   │        ├── analyze_data
           │         └─────────┘        └── take_note
           │
     ┌─────┴──────┐
     │  3 Tiers:  │
     │ • Window   │   (last 6 turns verbatim)
     │ • Summary  │   (older turns compressed)
     │ • Key Facts│   (never summarized away)
     └────────────┘

     Everything above is wrapped in:
     ┌─────────────────────────────────┐
     │       Tracer (trace.py)        │
     │  Every operation = a Span      │
     │  Spans nest into a tree        │
     │  Saved as JSONL + dashboard    │
     └─────────────────────────────────┘
```

### The Files and What They Do

| File | What it does | Analogy |
|------|-------------|---------|
| **`trace.py`** | Records everything that happens | A flight recorder / black box |
| **`llm.py`** | Talks to AI models (Gemini, OpenAI, Claude) | The brain's connection to the outside world |
| **`tools.py`** | Search, read, calculate, analyze, take notes | The agent's hands — things it can *do* |
| **`context.py`** | Decides what the AI sees each turn | A librarian picking which books to show |
| **`agent.py`** | The decision loop: plan → act → observe → repeat | The conductor of an orchestra |
| **`cli.py`** | Interactive chat terminal | The user interface |
| **`dashboard.py`** | Generates the visual HTML dashboard | The MRI scanner — see inside the brain |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+** installed
- **pip** (comes with Python)
- **No API keys needed** to run the eval suite (uses a deterministic stub)

### Installation

```bash
# Clone the repository
git clone https://github.com/ReasonableExcuses/prism.git
cd prism

# Install dependencies
pip install -r requirements.txt
```

### Run the Eval Suite (Proves Everything Works)

```bash
python evals/run_evals.py -v
```

Expected output:
```
════════════════════════════════════════════════════════════
  🔮 PRISM EVAL SUITE
════════════════════════════════════════════════════════════

  ✅ s01_basic_trace       — 5/5 assertions
  ✅ s02_tool_trace        — 3/3 assertions
  ✅ s03_nested_spans      — 4/4 assertions
  ✅ s04_failure_recovery  — 3/3 assertions
  ✅ s05_long_conversation — 23/23 assertions
  ✅ s06_context_budget    — 11/11 assertions
  ✅ s07_rolling_summary   — 2/2 assertions
  ✅ s08_cost_tracking     — 4/4 assertions
  ✅ s09_multi_tool        — 3/3 assertions
  ✅ s10_dashboard_gen     — 9/9 assertions

  ✅ ALL PASSED — Scenarios: 10/10, Assertions: 67/67
```

### Start the Interactive Web App (Streamlit)

```bash
streamlit run app.py
```
> Opens an interactive web interface at `http://localhost:8501`. Judges can chat with the agent live, toggle failure modes to see real-time error recovery, and watch the 7-tab dashboard (span tree, flamegraph, Plotly token flow, context budget bars, tool analytics, and conversation replay) update with every query.

### Start the Terminal Chat (CLI)

```bash
python prism/cli.py                      # deterministic mode (no API key)
python prism/cli.py --llm gemini         # Google Gemini (needs GEMINI_API_KEY)
python prism/cli.py --llm openai         # OpenAI GPT (needs OPENAI_API_KEY)
python prism/cli.py --llm anthropic      # Claude (needs ANTHROPIC_API_KEY)
```

### 🌐 Deployments (For Judges & Reviewers)

| Platform | Type | How to Access / Deploy |
|:---|:---|:---|
| **Streamlit Community Cloud** | Live Interactive Agent + Traces | Connect GitHub repo `ReasonableExcuses/prism`, select `app.py`, and click Deploy. Free hosting with live chat and real-time observability. |
| **Netlify** | Self-Contained Static Dashboard | Zero-config static deploy via `netlify.toml` and pre-built `public/index.html`. Connect the repo to Netlify or drag-and-drop the `public/` directory for instant hosting. |

To re-generate the static Netlify dashboard with fresh data:
```bash
python scripts/generate_demo_dashboard.py
```


---

## 🎬 Demo Walkthrough

Here's a 5-minute demo script. This is what you'd show a judge:

### Step 1: Start the REPL
```bash
python prism/cli.py
```

### Step 2: Ask a Research Question
```
you ❯ Search for Python programming language
```
The agent will:
1. **Plan** — decide to use `web_search`
2. **Act** — call the search tool
3. **Observe** — get results back
4. **Respond** — synthesize an answer

You'll see a footer: `3 steps | tools: web_search | 542 tokens | $0.000000 | 210ms`

### Step 3: Try a Calculation
```
you ❯ Calculate 25 * 37 + 100
```
The agent selects the `calculate` tool (not web_search!) — proving dynamic tool selection.

### Step 4: Inspect the Trace
```
you ❯ :trace
```
You'll see the full span tree — every operation, nested, with timings and token counts.

### Step 5: Check Costs
```
you ❯ :cost
```
Shows cumulative: LLM calls, tokens, cost breakdown.

### Step 6: Demo a Failure
```
you ❯ :fail
you ❯ Search for quantum computing
```
The search tool deliberately fails. The agent catches the error, reads the recovery hint, and handles it gracefully. The trace shows the failure → recovery arc.

```
you ❯ :unfail
```

### Step 7: Open the Dashboard 🎉
```
you ❯ :dashboard
```
This generates `artifacts/dashboard.html` and opens it in your browser. **This is the wow moment** — a dark-themed, interactive dashboard showing everything that happened.

---

## 📁 Project Structure

```
prism/                          ← You are here
│
├── prism/                      ← Core source code
│   ├── __init__.py             ← Package definition
│   ├── trace.py                ← 🔍 Span-level tracing system
│   ├── llm.py                  ← 🧠 LLM interface (4 providers)
│   ├── tools.py                ← 🔧 5 tools with error recovery
│   ├── context.py              ← 📦 3-tier context engineering
│   ├── agent.py                ← 🤖 Plan→Act→Observe loop
│   ├── cli.py                  ← 💻 Interactive REPL
│   └── dashboard.py            ← 📊 HTML dashboard generator
│
├── evals/                      ← Evaluation suite
│   ├── __init__.py
│   ├── harness.py              ← Test infrastructure
│   ├── scenarios.py            ← 10 test scenarios
│   └── run_evals.py            ← Runner script
│
├── app.py                      ← 🔮 Interactive Streamlit app with live tracing & chat
├── .streamlit/config.toml      ← Streamlit dark theme & styling config
├── public/index.html           ← 🌐 Pre-built static dashboard for Netlify hosting
├── netlify.toml                ← Netlify deployment configuration
├── scripts/                    ← Utility scripts (e.g. generate_demo_dashboard.py)
├── requirements.txt            ← Python dependencies (requests, bs4, rich, streamlit, plotly)
├── .gitignore                  ← Git ignore rules
└── README.md                   ← This file
```

### Reading Order (If You Want to Understand the Code)

1. **Start with** [`trace.py`](prism/trace.py) — ~260 lines. This is the foundation. Everything wraps itself in spans from this module.
2. **Then** [`llm.py`](prism/llm.py) — ~600 lines. See how one interface supports four different AI providers.
3. **Then** [`tools.py`](prism/tools.py) — ~380 lines. Five real tools with structured errors.
4. **Then** [`context.py`](prism/context.py) — ~280 lines. How the AI's "memory" is managed across long conversations.
5. **Then** [`agent.py`](prism/agent.py) — ~230 lines. The core loop that ties everything together.
6. **Finally** [`dashboard.py`](prism/dashboard.py) — ~500 lines. How all that trace data becomes a beautiful visualization.

---

## 🧪 The Eval Suite

Every eval scenario is designed to prove one specific claim. If any scenario fails, something is broken.

| Scenario | What It Proves | Why It Matters |
|----------|---------------|----------------|
| `s01_basic_trace` | Every LLM call produces a traced span with token counts | Core observability works |
| `s02_tool_trace` | Every tool call is traced with its inputs/outputs | Tool visibility works |
| `s03_nested_spans` | Spans nest correctly (agent → plan → LLM → tool) | The trace tree is accurate |
| `s04_failure_recovery` | When a tool fails, the agent catches it and recovers | Graceful error handling |
| `s05_long_conversation` | 20-turn conversation completes without crashing | Context engineering works |
| `s06_context_budget` | Token budget is never exceeded | Memory management works |
| `s07_rolling_summary` | Old turns are summarized, not just dropped | Information is preserved |
| `s08_cost_tracking` | Token/cost accumulates correctly across queries | Billing accuracy |
| `s09_multi_tool` | The agent picks different tools for different questions | Dynamic tool selection |
| `s10_dashboard_gen` | Dashboard HTML includes all 7 required sections | Visualization completeness |

Run a specific scenario:
```bash
python evals/run_evals.py -v --only s04_failure_recovery
```

---

## 📦 Context Engineering Explained

This is one of the most important (and hardest) parts of the project. Here's the problem:

> An LLM can only "see" a limited amount of text at once. In a 20-turn conversation, you can't send everything. So what do you cut?

### Our 3-Tier Strategy

```
┌──────────────────────────────────────────────────────┐
│                   CONTEXT WINDOW                      │
│                                                       │
│  ┌──────────────────────────────────────────────┐    │
│  │  TIER 1: System Prompt (500 tokens)          │    │
│  │  "You are Prism, an AI research assistant..." │    │
│  └──────────────────────────────────────────────┘    │
│                                                       │
│  ┌──────────────────────────────────────────────┐    │
│  │  TIER 2: Key Facts (200 tokens)              │    │
│  │  Critical info that must NEVER be lost        │    │
│  │  e.g., "User prefers Python over Java"       │    │
│  └──────────────────────────────────────────────┘    │
│                                                       │
│  ┌──────────────────────────────────────────────┐    │
│  │  TIER 3: Rolling Summary (400 tokens)         │    │
│  │  Compressed version of turns 1-14             │    │
│  │  "User asked about climate change, discussed  │    │
│  │   solar power, wind energy, and policy..."    │    │
│  └──────────────────────────────────────────────┘    │
│                                                       │
│  ┌──────────────────────────────────────────────┐    │
│  │  TIER 4: Recent Turns (800 tokens)            │    │
│  │  Turns 15-20 kept VERBATIM (word for word)   │    │
│  │  This is the "sliding window"                 │    │
│  └──────────────────────────────────────────────┘    │
│                                                       │
│  ┌──────────────────────────────────────────────┐    │
│  │  TIER 5: Response Room (1000 tokens)          │    │
│  │  Reserved space for the AI's answer           │    │
│  └──────────────────────────────────────────────┘    │
│                                                       │
│  Total Budget: 3,500 tokens                          │
└──────────────────────────────────────────────────────┘
```

Every time a turn is dropped from the window, it's compressed into the rolling summary. The dashboard shows this as a stacked bar chart — you can *see* the context allocation change over time.

---

## 🔌 LLM Provider Support

Prism supports four "backends" for the AI brain:

| Provider | Model | API Key | Notes |
|----------|-------|---------|-------|
| **Deterministic** (default) | `deterministic` | None needed | Rule-based stub for reproducible testing |
| **Google Gemini** | `gemini-2.5-flash` | `GEMINI_API_KEY` | Fast and cost-effective |
| **OpenAI** | `gpt-4o-mini` | `OPENAI_API_KEY` | Industry standard |
| **Anthropic Claude** | `claude-haiku-3.5` | `ANTHROPIC_API_KEY` | Strong reasoning |

### How Provider Selection Works

```
1. If you pass --llm gemini       → uses Gemini
2. If GEMINI_API_KEY is set        → auto-detects Gemini
3. If OPENAI_API_KEY is set        → auto-detects OpenAI
4. If ANTHROPIC_API_KEY is set     → auto-detects Claude
5. If nothing is set               → uses DeterministicLLM (free, offline)
```

The `DeterministicLLM` is not a real AI — it's a rule-based system that pattern-matches keywords to tools. Its purpose is making the eval suite reproducible: it always gives the exact same output for the same input, so tests measure the *agent logic*, not the model's randomness.

---

## 🏆 How We Meet the Judging Criteria

| Criterion | Weight | How Prism Addresses It |
|-----------|--------|------------------------|
| **Problem Depth** | 25% | Five real tools, 3-tier context engineering, structured error recovery, multi-provider LLM support |
| **Technical Execution** | 25% | 10/10 eval scenarios, 67/67 assertions, ~9.5s offline, no framework dependencies |
| **Rule Compliance** | 20% | Trace/log every run ✅ · Failure case caught via tracing ✅ · 20-turn test ✅ · Token/cost tracked ✅ |
| **Edge Case Handling** | 15% | Tool failures (s04), long conversations (s05), budget enforcement (s06), dynamic tool selection (s09), no-data honesty (s12) |
| **Explanation & Demo** | 15% | `:trace` / `:cost` / `:context` / `:dashboard` in the REPL; this README; every module header explains its thesis |

---

## ⚠️ Known Limitations

We believe in being honest about what the system can and cannot do:

1. **Default mode uses a stub, not a real LLM.** `DeterministicLLM` handles simple keyword matching well but will miss nuanced phrasing. This is by design — it makes the eval suite reproducible. Use `--llm gemini` (or openai/anthropic) for the real experience.

2. **Web search uses DuckDuckGo's instant answer API.** It's free and requires no API key, but returns limited results compared to a full search API. For a production system, you'd swap in SerpAPI or Tavily.

3. **Cost shows $0.0000 in deterministic mode.** The plumbing is fully exercised — every span tracks tokens and cost — but the stub is priced at zero. With a real API key, it reports real pricing.

4. **Embeddings are not used.** This is a pure agent project (Track 1), not a RAG system. There's no vector database. The context engineering is based on sliding windows and summarization, not semantic retrieval.

5. **The dashboard requires a browser.** It's a self-contained HTML file with embedded CSS and JS. No server needed, but it won't render in a terminal.

---

## 🤝 Team

Built by **Team ReasonableExcuses** for Epochesque 2.0.

---

## 📜 License

MIT License — use it, learn from it, build on it.
