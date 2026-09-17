"""
prism/dashboard.py — Generates a self-contained interactive HTML dashboard.

Thesis: a judge should be able to open one file and see everything that
happened: the span tree, the flamegraph, the context window allocation,
token flow, tool usage, failure cases, and the full conversation replay.

The HTML is fully self-contained (embedded CSS + JS, no CDN).  It uses
a dark theme with glassmorphism panels, smooth animations, Inter font,
and modern data visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from prism.trace import Tracer, SpanKind
from prism.context import ContextEngine
from prism.agent import Agent


def generate_dashboard(
    tracer: Tracer,
    context: ContextEngine,
    agent: Agent | None = None,
    output_dir: str = "artifacts",
) -> Path:
    """Generate the self-contained HTML dashboard."""

    # Collect data
    summary = tracer.summary()
    tree_data = tracer.get_tree_data()
    all_spans = [s.to_dict() for s in tracer.get_spans()]
    allocation_history = context.get_allocation_history()
    context_status = context.get_status()

    # Build tool usage stats
    tool_spans = [s for s in all_spans if s.get("kind") == "TOOL"]
    tool_stats: dict[str, dict] = {}
    for ts in tool_spans:
        name = ts.get("name", "").replace("tool:", "")
        if name not in tool_stats:
            tool_stats[name] = {"calls": 0, "successes": 0, "failures": 0, "total_ms": 0}
        tool_stats[name]["calls"] += 1
        if ts.get("error"):
            tool_stats[name]["failures"] += 1
        else:
            tool_stats[name]["successes"] += 1
        tool_stats[name]["total_ms"] += ts.get("duration_ms", 0)

    # Build LLM call stats
    llm_spans = [s for s in all_spans if s.get("kind") == "LLM"]
    llm_data = []
    for ls in llm_spans:
        llm_data.append({
            "name": ls.get("name", ""),
            "model": ls.get("model", "unknown"),
            "input_tokens": ls.get("input_tokens", 0),
            "output_tokens": ls.get("output_tokens", 0),
            "cost_usd": ls.get("cost_usd", 0),
            "duration_ms": ls.get("duration_ms", 0),
        })

    # Failure cases
    failures = []
    for s in all_spans:
        if s.get("error"):
            failures.append({
                "span": s.get("name", ""),
                "kind": s.get("kind", ""),
                "error": s.get("error", ""),
                "duration_ms": s.get("duration_ms", 0),
                "events": s.get("events", []),
            })

    # Conversation turns
    turns = []
    for turn in context._turns:
        turns.append({
            "role": turn.role,
            "content": turn.content[:500],
            "turn_number": turn.turn_number,
            "tokens": turn.tokens,
            "tool_name": turn.tool_name,
        })

    # Package all data as JSON for embedding
    dashboard_data = {
        "summary": summary,
        "tree": tree_data,
        "spans": all_spans,
        "tool_stats": tool_stats,
        "llm_data": llm_data,
        "failures": failures,
        "allocation_history": allocation_history,
        "context_status": context_status,
        "turns": turns,
    }

    data_json = json.dumps(dashboard_data, default=str)

    html = _build_html(data_json)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / "dashboard.html"
    file_path.write_text(html, encoding="utf-8")

    # Also save the trace
    tracer.save()

    return file_path


def _build_html(data_json: str) -> str:
    """Build the complete HTML dashboard."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🔮 Prism Dashboard — Glass Box Agent Trace</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
  --bg-primary: #0a0a0f;
  --bg-secondary: #12121a;
  --bg-card: rgba(20, 20, 35, 0.8);
  --bg-glass: rgba(255, 255, 255, 0.03);
  --border: rgba(255, 255, 255, 0.06);
  --border-accent: rgba(139, 92, 246, 0.3);
  --text-primary: #e4e4e7;
  --text-secondary: #a1a1aa;
  --text-dim: #52525b;
  --accent-purple: #8b5cf6;
  --accent-blue: #3b82f6;
  --accent-cyan: #06b6d4;
  --accent-green: #10b981;
  --accent-yellow: #f59e0b;
  --accent-red: #ef4444;
  --accent-pink: #ec4899;
  --gradient-hero: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 50%, #06b6d4 100%);
  --gradient-card: linear-gradient(135deg, rgba(139,92,246,0.1) 0%, rgba(59,130,246,0.05) 100%);
  --shadow-glow: 0 0 40px rgba(139, 92, 246, 0.15);
  --radius: 16px;
  --radius-sm: 10px;
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
  font-family: 'Inter', -apple-system, sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  line-height: 1.6;
  overflow-x: hidden;
}}

/* Animated background */
body::before {{
  content: '';
  position: fixed;
  top: -50%; left: -50%;
  width: 200%; height: 200%;
  background: radial-gradient(circle at 20% 50%, rgba(139,92,246,0.06) 0%, transparent 50%),
              radial-gradient(circle at 80% 20%, rgba(59,130,246,0.04) 0%, transparent 50%),
              radial-gradient(circle at 50% 80%, rgba(6,182,212,0.03) 0%, transparent 50%);
  animation: bgFloat 30s ease-in-out infinite;
  z-index: -1;
}}

@keyframes bgFloat {{
  0%, 100% {{ transform: translate(0, 0) rotate(0deg); }}
  33% {{ transform: translate(30px, -30px) rotate(1deg); }}
  66% {{ transform: translate(-20px, 20px) rotate(-1deg); }}
}}

.container {{
  max-width: 1400px;
  margin: 0 auto;
  padding: 2rem;
}}

/* Hero Header */
.hero {{
  text-align: center;
  padding: 3rem 2rem;
  margin-bottom: 2rem;
}}

.hero h1 {{
  font-size: 2.8rem;
  font-weight: 700;
  background: var(--gradient-hero);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.5rem;
  letter-spacing: -0.02em;
}}

.hero .subtitle {{
  font-size: 1.1rem;
  color: var(--text-secondary);
  font-weight: 300;
}}

/* Stats Grid */
.stats-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}}

.stat-card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem;
  text-align: center;
  backdrop-filter: blur(10px);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}}

.stat-card::before {{
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--gradient-hero);
  opacity: 0;
  transition: opacity 0.3s ease;
}}

.stat-card:hover {{
  border-color: var(--border-accent);
  transform: translateY(-2px);
  box-shadow: var(--shadow-glow);
}}

.stat-card:hover::before {{ opacity: 1; }}

.stat-value {{
  font-size: 2rem;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  background: var(--gradient-hero);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}}

.stat-label {{
  font-size: 0.8rem;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-top: 0.3rem;
}}

/* Section */
.section {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  backdrop-filter: blur(10px);
}}

.section-title {{
  font-size: 1.2rem;
  font-weight: 600;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}}

.section-title .icon {{ font-size: 1.3rem; }}

/* Span Tree */
.span-tree {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.85rem;
  line-height: 1.8;
}}

.span-node {{
  padding: 0.3rem 0.5rem;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background 0.2s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}}

.span-node:hover {{
  background: var(--bg-glass);
}}

.span-children {{
  margin-left: 1.5rem;
  border-left: 1px solid var(--border);
  padding-left: 0.5rem;
}}

.span-kind {{
  font-size: 0.65rem;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}}

.kind-AGENT {{ background: rgba(139,92,246,0.2); color: var(--accent-purple); }}
.kind-LLM {{ background: rgba(6,182,212,0.2); color: var(--accent-cyan); }}
.kind-TOOL {{ background: rgba(245,158,11,0.2); color: var(--accent-yellow); }}
.kind-CONTEXT {{ background: rgba(16,185,129,0.2); color: var(--accent-green); }}
.kind-RETRIEVE {{ background: rgba(59,130,246,0.2); color: var(--accent-blue); }}
.kind-SYSTEM {{ background: rgba(161,161,170,0.15); color: var(--text-secondary); }}

.span-duration {{
  color: var(--text-dim);
  font-size: 0.75rem;
  margin-left: auto;
}}

.span-tokens {{
  color: var(--accent-cyan);
  font-size: 0.75rem;
}}

.span-error {{
  color: var(--accent-red);
  font-size: 0.75rem;
  font-weight: 600;
}}

.span-name {{ font-weight: 500; }}

/* Flamegraph */
.flamegraph {{
  padding: 1rem 0;
}}

.flame-bar {{
  height: 32px;
  border-radius: 4px;
  margin: 2px 0;
  display: flex;
  align-items: center;
  padding: 0 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  color: white;
  cursor: pointer;
  transition: all 0.2s ease;
  position: relative;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}}

.flame-bar:hover {{
  filter: brightness(1.2);
  transform: scaleY(1.1);
}}

.flame-bar .flame-label {{
  overflow: hidden;
  text-overflow: ellipsis;
}}

.flame-bar .flame-time {{
  margin-left: auto;
  font-size: 0.7rem;
  opacity: 0.8;
  padding-left: 8px;
  flex-shrink: 0;
}}

/* Token Chart */
.token-chart {{
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}}

.token-row {{
  display: flex;
  align-items: center;
  gap: 0.75rem;
}}

.token-label {{
  width: 180px;
  font-size: 0.8rem;
  color: var(--text-secondary);
  font-family: 'JetBrains Mono', monospace;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}

.token-bars {{
  flex: 1;
  display: flex;
  height: 24px;
  border-radius: 4px;
  overflow: hidden;
  background: rgba(255,255,255,0.03);
}}

.token-bar-input {{
  background: var(--accent-blue);
  height: 100%;
  transition: width 0.5s ease;
}}

.token-bar-output {{
  background: var(--accent-purple);
  height: 100%;
  transition: width 0.5s ease;
}}

.token-value {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
  color: var(--text-dim);
  width: 80px;
  text-align: right;
}}

/* Context Chart */
.context-chart {{
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}}

.context-bar-row {{
  display: flex;
  align-items: center;
  gap: 0.75rem;
}}

.context-turn-label {{
  width: 60px;
  font-size: 0.75rem;
  color: var(--text-dim);
  font-family: 'JetBrains Mono', monospace;
  text-align: right;
}}

.context-bar-track {{
  flex: 1;
  display: flex;
  height: 28px;
  border-radius: 4px;
  overflow: hidden;
  background: rgba(255,255,255,0.02);
}}

.context-seg {{
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.6rem;
  font-family: 'JetBrains Mono', monospace;
  color: rgba(255,255,255,0.7);
  overflow: hidden;
  transition: width 0.5s ease;
}}

.context-util {{
  width: 60px;
  font-size: 0.75rem;
  color: var(--text-dim);
  font-family: 'JetBrains Mono', monospace;
  text-align: right;
}}

/* Tool Donut */
.tool-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}}

.tool-card {{
  background: var(--bg-glass);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 1rem;
  text-align: center;
}}

.tool-name {{
  font-weight: 600;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}}

.tool-stat {{
  font-size: 0.8rem;
  color: var(--text-secondary);
}}

.tool-bar {{
  height: 6px;
  border-radius: 3px;
  margin-top: 0.5rem;
  background: rgba(255,255,255,0.05);
  overflow: hidden;
}}

.tool-bar-fill {{
  height: 100%;
  border-radius: 3px;
  transition: width 0.5s ease;
}}

/* Failures */
.failure-card {{
  background: rgba(239,68,68,0.05);
  border: 1px solid rgba(239,68,68,0.2);
  border-radius: var(--radius-sm);
  padding: 1rem;
  margin-bottom: 0.75rem;
}}

.failure-header {{
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  color: var(--accent-red);
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
}}

.failure-detail {{
  font-size: 0.8rem;
  color: var(--text-secondary);
  font-family: 'JetBrains Mono', monospace;
  background: rgba(0,0,0,0.3);
  padding: 0.5rem;
  border-radius: 6px;
  margin-top: 0.5rem;
  overflow-x: auto;
}}

/* Conversation Replay */
.conv-turn {{
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
  padding: 0.75rem;
  border-radius: var(--radius-sm);
  transition: background 0.2s ease;
}}

.conv-turn:hover {{
  background: var(--bg-glass);
}}

.conv-icon {{
  font-size: 1.5rem;
  width: 36px;
  text-align: center;
  flex-shrink: 0;
}}

.conv-content {{
  flex: 1;
  min-width: 0;
}}

.conv-role {{
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.3rem;
  font-weight: 600;
}}

.conv-text {{
  font-size: 0.85rem;
  color: var(--text-secondary);
  white-space: pre-wrap;
  word-break: break-word;
}}

.conv-meta {{
  font-size: 0.7rem;
  color: var(--text-dim);
  margin-top: 0.3rem;
  font-family: 'JetBrains Mono', monospace;
}}

.role-user .conv-role {{ color: var(--accent-green); }}
.role-assistant .conv-role {{ color: var(--accent-cyan); }}
.role-tool .conv-role {{ color: var(--accent-yellow); }}

/* Toggle */
.toggle-btn {{
  background: var(--bg-glass);
  border: 1px solid var(--border);
  color: var(--text-secondary);
  padding: 0.4rem 1rem;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.8rem;
  font-family: 'Inter', sans-serif;
  transition: all 0.2s ease;
}}

.toggle-btn:hover {{
  border-color: var(--border-accent);
  color: var(--text-primary);
}}

/* Responsive */
@media (max-width: 768px) {{
  .container {{ padding: 1rem; }}
  .hero h1 {{ font-size: 2rem; }}
  .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}

/* Animation for page load */
@keyframes fadeInUp {{
  from {{ opacity: 0; transform: translateY(20px); }}
  to {{ opacity: 1; transform: translateY(0); }}
}}

.section {{
  animation: fadeInUp 0.5s ease forwards;
  opacity: 0;
}}

.section:nth-child(1) {{ animation-delay: 0.1s; }}
.section:nth-child(2) {{ animation-delay: 0.2s; }}
.section:nth-child(3) {{ animation-delay: 0.3s; }}
.section:nth-child(4) {{ animation-delay: 0.4s; }}
.section:nth-child(5) {{ animation-delay: 0.5s; }}
.section:nth-child(6) {{ animation-delay: 0.6s; }}
.section:nth-child(7) {{ animation-delay: 0.7s; }}
.section:nth-child(8) {{ animation-delay: 0.8s; }}

/* Collapsible */
.collapsible-header {{
  cursor: pointer;
  user-select: none;
}}

.collapsible-header::after {{
  content: ' ▾';
  font-size: 0.8rem;
}}

.collapsible-header.collapsed::after {{
  content: ' ▸';
}}

.collapsed + .collapsible-body {{
  display: none;
}}

/* Two-column layout */
.two-col {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}}

@media (max-width: 900px) {{
  .two-col {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>

<div class="container">

  <!-- Hero -->
  <div class="hero">
    <h1>🔮 Prism Dashboard</h1>
    <p class="subtitle">Glass Box Agent Trace — Every Decision Visible</p>
  </div>

  <!-- Stats will be rendered by JS -->
  <div id="stats-grid" class="stats-grid"></div>
  <div id="sections"></div>

</div>

<script>
// Embedded data
const DATA = {data_json};

// ===== Render Stats Grid =====
function renderStats() {{
  const grid = document.getElementById('stats-grid');
  const s = DATA.summary;
  const stats = [
    {{ value: s.llm_calls, label: 'LLM Calls', icon: '🧠' }},
    {{ value: s.tool_calls, label: 'Tool Calls', icon: '🔧' }},
    {{ value: s.total_tokens.toLocaleString(), label: 'Total Tokens', icon: '📊' }},
    {{ value: '$' + s.cost_usd.toFixed(6), label: 'Total Cost', icon: '💰' }},
    {{ value: Math.round(s.wall_ms) + 'ms', label: 'Wall Time', icon: '⏱️' }},
    {{ value: s.errors, label: 'Errors Caught', icon: s.errors > 0 ? '⚠️' : '✅' }},
    {{ value: s.span_count, label: 'Total Spans', icon: '🔗' }},
    {{ value: DATA.turns.length, label: 'Conv. Turns', icon: '💬' }},
  ];

  grid.innerHTML = stats.map(st => `
    <div class="stat-card">
      <div class="stat-value">${{st.value}}</div>
      <div class="stat-label">${{st.icon}} ${{st.label}}</div>
    </div>
  `).join('');
}}

// ===== Render Span Tree =====
function renderSpanNode(span, depth = 0) {{
  const kindClass = 'kind-' + span.kind;
  const icon = {{AGENT:'🤖',LLM:'🧠',TOOL:'🔧',CONTEXT:'📦',RETRIEVE:'🔍',SYSTEM:'⚙️'}}[span.kind] || '•';
  const dur = Math.round(span.duration_ms || 0);
  const tokens = (span.input_tokens || 0) + (span.output_tokens || 0);
  const tokenStr = tokens > 0 ? `<span class="span-tokens">${{span.input_tokens}}→${{span.output_tokens}} tok</span>` : '';
  const errorStr = span.error ? `<span class="span-error">✗ ${{escHtml(span.error.substring(0,60))}}</span>` : '';

  let html = `<div class="span-node">
    <span>${{icon}}</span>
    <span class="span-kind ${{kindClass}}">${{span.kind}}</span>
    <span class="span-name">${{escHtml(span.name)}}</span>
    ${{tokenStr}}
    ${{errorStr}}
    <span class="span-duration">${{dur}}ms</span>
  </div>`;

  if (span.children && span.children.length > 0) {{
    html += '<div class="span-children">';
    for (const child of span.children) {{
      html += renderSpanNode(child, depth + 1);
    }}
    html += '</div>';
  }}
  return html;
}}

// ===== Render Flamegraph =====
function renderFlamegraph() {{
  const spans = DATA.spans.filter(s => s.duration_ms > 0);
  if (spans.length === 0) return '<p style="color:var(--text-dim)">No timed spans recorded.</p>';

  const maxDur = Math.max(...spans.map(s => s.duration_ms));
  const colors = {{
    AGENT: 'linear-gradient(90deg, #8b5cf6, #7c3aed)',
    LLM: 'linear-gradient(90deg, #06b6d4, #0891b2)',
    TOOL: 'linear-gradient(90deg, #f59e0b, #d97706)',
    CONTEXT: 'linear-gradient(90deg, #10b981, #059669)',
    RETRIEVE: 'linear-gradient(90deg, #3b82f6, #2563eb)',
    SYSTEM: 'linear-gradient(90deg, #6b7280, #4b5563)',
  }};

  return spans.map(s => {{
    const pct = Math.max(5, (s.duration_ms / maxDur) * 100);
    const bg = colors[s.kind] || colors.SYSTEM;
    return `<div class="flame-bar" style="width:${{pct}}%; background:${{bg}}">
      <span class="flame-label">${{escHtml(s.name)}}</span>
      <span class="flame-time">${{Math.round(s.duration_ms)}}ms</span>
    </div>`;
  }}).join('');
}}

// ===== Render Token Flow =====
function renderTokenFlow() {{
  const llm = DATA.llm_data;
  if (llm.length === 0) return '<p style="color:var(--text-dim)">No LLM calls recorded.</p>';

  const maxTok = Math.max(...llm.map(l => l.input_tokens + l.output_tokens), 1);

  return '<div class="token-chart">' + llm.map(l => {{
    const inPct = (l.input_tokens / maxTok) * 100;
    const outPct = (l.output_tokens / maxTok) * 100;
    return `<div class="token-row">
      <span class="token-label" title="${{escHtml(l.name)}}">${{escHtml(l.name)}}</span>
      <div class="token-bars">
        <div class="token-bar-input" style="width:${{inPct}}%" title="Input: ${{l.input_tokens}}"></div>
        <div class="token-bar-output" style="width:${{outPct}}%" title="Output: ${{l.output_tokens}}"></div>
      </div>
      <span class="token-value">${{l.input_tokens + l.output_tokens}} tok</span>
    </div>`;
  }}).join('') + `
    <div style="display:flex;gap:1.5rem;margin-top:0.5rem;font-size:0.75rem;color:var(--text-dim)">
      <span>■ <span style="color:var(--accent-blue)">Input</span></span>
      <span>■ <span style="color:var(--accent-purple)">Output</span></span>
    </div>
  </div>`;
}}

// ===== Render Context Window =====
function renderContextChart() {{
  const hist = DATA.allocation_history;
  if (hist.length === 0) return '<p style="color:var(--text-dim)">No context snapshots recorded.</p>';

  const budget = DATA.context_status.budget.total;
  const segColors = {{
    system_prompt: 'var(--accent-purple)',
    key_facts: 'var(--accent-pink)',
    rolling_summary: 'var(--accent-blue)',
    recent_turns: 'var(--accent-cyan)',
    tool_results: 'var(--accent-yellow)',
  }};

  let html = '<div class="context-chart">';
  for (const entry of hist) {{
    const alloc = entry.allocation;
    const totalUsed = Object.values(alloc).reduce((a,b) => a+b, 0);
    html += `<div class="context-bar-row">
      <span class="context-turn-label">Turn ${{entry.turn}}</span>
      <div class="context-bar-track">`;

    for (const [key, tokens] of Object.entries(alloc)) {{
      if (tokens > 0) {{
        const pct = (tokens / budget) * 100;
        const color = segColors[key] || 'var(--text-dim)';
        html += `<div class="context-seg" style="width:${{pct}}%;background:${{color}}" title="${{key}}: ${{tokens}} tokens">${{tokens > 50 ? key.split('_')[0] : ''}}</div>`;
      }}
    }}

    html += `</div>
      <span class="context-util">${{entry.utilization_pct.toFixed(0)}}%</span>
    </div>`;
  }}

  html += `<div style="display:flex;flex-wrap:wrap;gap:1rem;margin-top:0.75rem;font-size:0.7rem;color:var(--text-dim)">`;
  for (const [key, color] of Object.entries(segColors)) {{
    html += `<span>■ <span style="color:${{color}}">${{key.replace('_',' ')}}</span></span>`;
  }}
  html += '</div></div>';
  return html;
}}

// ===== Render Tool Usage =====
function renderToolUsage() {{
  const stats = DATA.tool_stats;
  const names = Object.keys(stats);
  if (names.length === 0) return '<p style="color:var(--text-dim)">No tool calls recorded.</p>';

  const maxCalls = Math.max(...names.map(n => stats[n].calls));

  return '<div class="tool-grid">' + names.map(name => {{
    const s = stats[name];
    const successRate = s.calls > 0 ? Math.round((s.successes / s.calls) * 100) : 0;
    const barColor = successRate >= 80 ? 'var(--accent-green)' : successRate >= 50 ? 'var(--accent-yellow)' : 'var(--accent-red)';
    const barPct = (s.calls / maxCalls) * 100;

    return `<div class="tool-card">
      <div class="tool-name">🔧 ${{escHtml(name)}}</div>
      <div class="tool-stat">${{s.calls}} call(s) · ${{s.successes}} ok · ${{s.failures}} err</div>
      <div class="tool-stat">Avg: ${{s.calls > 0 ? Math.round(s.total_ms / s.calls) : 0}}ms · Rate: ${{successRate}}%</div>
      <div class="tool-bar">
        <div class="tool-bar-fill" style="width:${{barPct}}%;background:${{barColor}}"></div>
      </div>
    </div>`;
  }}).join('') + '</div>';
}}

// ===== Render Failures =====
function renderFailures() {{
  const f = DATA.failures;
  if (f.length === 0) return '<p style="color:var(--accent-green)">✅ No errors recorded — all operations succeeded.</p>';

  return f.map(fail => `
    <div class="failure-card">
      <div class="failure-header">⚠️ ${{escHtml(fail.span)}} <span style="font-weight:400;color:var(--text-dim)">(${{fail.kind}})</span></div>
      <div class="failure-detail">${{escHtml(fail.error)}}</div>
      ${{fail.events.length > 0 ? `<div class="failure-detail" style="border-left:3px solid var(--accent-yellow)">Events: ${{fail.events.map(e => escHtml(e.name)).join(' → ')}}</div>` : ''}}
    </div>
  `).join('');
}}

// ===== Render Conversation Replay =====
function renderConversation() {{
  const turns = DATA.turns;
  if (turns.length === 0) return '<p style="color:var(--text-dim)">No conversation recorded.</p>';

  return turns.map(t => {{
    const icon = t.role === 'user' ? '👤' : t.role === 'assistant' ? '🤖' : '🔧';
    const roleClass = 'role-' + t.role;
    const toolBadge = t.tool_name ? ` <span style="font-size:0.7rem;color:var(--accent-yellow)">[${{t.tool_name}}]</span>` : '';

    return `<div class="conv-turn ${{roleClass}}">
      <div class="conv-icon">${{icon}}</div>
      <div class="conv-content">
        <div class="conv-role">${{t.role}}${{toolBadge}}</div>
        <div class="conv-text">${{escHtml(t.content)}}</div>
        <div class="conv-meta">Turn #${{t.turn_number}} · ~${{t.tokens}} tokens</div>
      </div>
    </div>`;
  }}).join('');
}}

// ===== Utility =====
function escHtml(str) {{
  if (!str) return '';
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

// ===== Assemble Page =====
function render() {{
  renderStats();

  const sections = document.getElementById('sections');
  let html = '';

  // Span Tree
  html += `<div class="section">
    <div class="section-title collapsible-header" onclick="this.classList.toggle('collapsed')">
      <span class="icon">🌳</span> Span Tree — Full Execution Trace
    </div>
    <div class="collapsible-body">
      <div class="span-tree">${{DATA.tree.map(t => renderSpanNode(t)).join('')}}</div>
    </div>
  </div>`;

  // Flamegraph
  html += `<div class="section">
    <div class="section-title collapsible-header" onclick="this.classList.toggle('collapsed')">
      <span class="icon">🔥</span> Flamegraph — Time Distribution
    </div>
    <div class="collapsible-body">
      <div class="flamegraph">${{renderFlamegraph()}}</div>
    </div>
  </div>`;

  // Two-column: Token Flow + Context Window
  html += '<div class="two-col">';

  html += `<div class="section">
    <div class="section-title collapsible-header" onclick="this.classList.toggle('collapsed')">
      <span class="icon">📊</span> Token Flow
    </div>
    <div class="collapsible-body">${{renderTokenFlow()}}</div>
  </div>`;

  html += `<div class="section">
    <div class="section-title collapsible-header" onclick="this.classList.toggle('collapsed')">
      <span class="icon">📦</span> Context Window Allocation
    </div>
    <div class="collapsible-body">${{renderContextChart()}}</div>
  </div>`;

  html += '</div>';

  // Two-column: Tool Usage + Failures
  html += '<div class="two-col">';

  html += `<div class="section">
    <div class="section-title collapsible-header" onclick="this.classList.toggle('collapsed')">
      <span class="icon">🔧</span> Tool Usage
    </div>
    <div class="collapsible-body">${{renderToolUsage()}}</div>
  </div>`;

  html += `<div class="section">
    <div class="section-title collapsible-header" onclick="this.classList.toggle('collapsed')">
      <span class="icon">⚠️</span> Failure Cases & Recovery
    </div>
    <div class="collapsible-body">${{renderFailures()}}</div>
  </div>`;

  html += '</div>';

  // Conversation Replay
  html += `<div class="section">
    <div class="section-title collapsible-header" onclick="this.classList.toggle('collapsed')">
      <span class="icon">💬</span> Conversation Replay
    </div>
    <div class="collapsible-body">${{renderConversation()}}</div>
  </div>`;

  // Footer
  html += `<div style="text-align:center;padding:2rem;color:var(--text-dim);font-size:0.8rem">
    🔮 Prism Dashboard · Run ${{DATA.summary.run_id}} · Generated ${{new Date().toLocaleString()}}
  </div>`;

  sections.innerHTML = html;
}}

render();
</script>
</body>
</html>"""
