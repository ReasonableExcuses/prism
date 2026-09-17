"""
evals/scenarios.py — 10 scenarios that prove every claim in the README.

Each scenario is the shortest test that could falsify one claim.
All run with DeterministicLLM — no API key, reproducible to the byte.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from evals.harness import Harness


# ===== s01: Basic Trace =====
def s01_basic_trace(h: Harness) -> None:
    """Every LLM call produces a trace span with tokens and cost."""
    h.setup()
    h.query("Hello, how are you?")

    # There should be spans
    spans = h.tracer.get_spans()
    h.assert_gt(len(spans), 0, "Should have at least one span")

    # There should be LLM spans
    llm_spans = [s for s in spans if s.kind.value == "LLM"]
    h.assert_gt(len(llm_spans), 0, "Should have at least one LLM span")

    # LLM spans should have token counts
    for s in llm_spans:
        h.assert_gt(s.input_tokens + s.output_tokens, 0,
                     f"LLM span '{s.name}' should have tokens")

    # Summary should have correct rollups
    summary = h.tracer.summary()
    h.assert_gt(summary["llm_calls"], 0, "Summary should count LLM calls")
    h.assert_gt(summary["total_tokens"], 0, "Summary should count total tokens")


# ===== s02: Tool Trace =====
def s02_tool_trace(h: Harness) -> None:
    """Every tool call produces a trace span with input/output."""
    h.setup()
    h.query("Search for Python programming language")

    spans = h.tracer.get_spans()
    tool_spans = [s for s in spans if s.kind.value == "TOOL"]
    h.assert_gt(len(tool_spans), 0, "Should have at least one tool span")

    for s in tool_spans:
        h.assert_true("tool_name" in s.attrs, f"Tool span should have tool_name attr")
        h.assert_true("arguments" in s.attrs, f"Tool span should have arguments attr")


# ===== s03: Nested Spans =====
def s03_nested_spans(h: Harness) -> None:
    """Spans are properly nested: agent → plan → tool → etc."""
    h.setup()
    h.query("Search for quantum computing")

    root_spans = h.tracer.get_root_spans()
    h.assert_gt(len(root_spans), 0, "Should have root spans")

    # Root span should be AGENT
    root = root_spans[0]
    h.assert_eq(root.kind.value, "AGENT", "Root span should be AGENT kind")

    # Root should have children
    h.assert_gt(len(root.children), 0, "Agent span should have child spans")

    # Check that the tree has depth
    def max_depth(span, d=0):
        if not span.children:
            return d
        return max(max_depth(c, d + 1) for c in span.children)

    depth = max_depth(root)
    h.assert_gt(depth, 1, f"Span tree should have depth > 1, got {depth}")


# ===== s04: Failure Recovery =====
def s04_failure_recovery(h: Harness) -> None:
    """Tool failure is caught, traced, and the agent recovers."""
    h.setup(fail_search=True)
    response = h.query("Search for information about Mars")

    # The response should still exist (agent didn't crash)
    h.assert_true(len(response) > 0, "Agent should produce a response even when tools fail")

    # There should be error spans
    spans = h.tracer.get_spans()
    error_spans = [s for s in spans if s.error]
    h.assert_gt(len(error_spans), 0, "Should have error spans when tools fail")

    # The tracer should count errors
    h.assert_gt(h.tracer.total_errors, 0, "Error count should be > 0")


# ===== s05: Long Conversation =====
def s05_long_conversation(h: Harness) -> None:
    """20-turn conversation doesn't break; context is managed."""
    h.setup()

    queries = [
        "Hello, I'm researching climate change.",
        "What are the main causes?",
        "Tell me about carbon emissions.",
        "What about deforestation?",
        "How does it affect ocean levels?",
        "What is the Paris Agreement?",
        "Which countries emit the most?",
        "What are renewable energy options?",
        "Tell me about solar power.",
        "What about wind energy?",
        "How effective is nuclear power?",
        "What is carbon capture?",
        "Tell me about electric vehicles.",
        "What are the economic impacts?",
        "How does it affect agriculture?",
        "What about biodiversity loss?",
        "Tell me about climate adaptation.",
        "What can individuals do?",
        "What role does policy play?",
        "Summarize our conversation.",
    ]

    responses = []
    for q in queries:
        resp = h.query(q)
        responses.append(resp)
        h.assert_true(len(resp) > 0, f"Turn should produce non-empty response: '{q[:30]}'")

    # All 20 turns should have completed
    h.assert_eq(len(responses), 20, "All 20 turns should complete")

    # Context engine should have recorded turns
    h.assert_gt(h.context.turn_count, 30, "Should have many turns recorded (user + assistant)")

    # The context engine should be summarizing (window < total)
    status = h.context.get_status()
    h.assert_gt(status["total_turns"], status["window_size"],
                "Total turns should exceed window size, triggering summarization")


# ===== s06: Context Budget =====
def s06_context_budget(h: Harness) -> None:
    """Token budget is respected at every turn."""
    h.setup()

    # Add several turns to fill the window
    for i in range(10):
        h.query(f"Tell me fact number {i + 1} about space exploration and NASA missions.")

    # Check allocation history
    history = h.context.get_allocation_history()
    h.assert_gt(len(history), 0, "Should have allocation history")

    for entry in history:
        total_allocated = sum(entry["allocation"].values())
        budget_total = h.context.budget.total
        h.assert_true(
            total_allocated <= budget_total,
            f"Allocated {total_allocated} tokens should be <= budget {budget_total}"
        )


# ===== s07: Rolling Summary =====
def s07_rolling_summary(h: Harness) -> None:
    """Older turns are summarized; the summary captures key info."""
    h.setup()

    # Fill beyond window size
    for i in range(10):
        h.query(f"Question {i + 1}: What is {['Python', 'Java', 'Go', 'Rust', 'C++', 'Ruby', 'Swift', 'Kotlin', 'TypeScript', 'Scala'][i]}?")

    status = h.context.get_status()
    h.assert_true(status["has_summary"], "Should have a rolling summary after many turns")
    h.assert_gt(status["summary_length"], 0, "Summary should have content")


# ===== s08: Cost Tracking =====
def s08_cost_tracking(h: Harness) -> None:
    """Cumulative cost is accurately tracked across the run."""
    h.setup()

    h.query("Hello")
    cost_1 = h.tracer.total_cost_usd
    tokens_1 = h.tracer.total_tokens

    h.query("Search for AI news")
    cost_2 = h.tracer.total_cost_usd
    tokens_2 = h.tracer.total_tokens

    # Cost should be non-negative
    h.assert_true(cost_1 >= 0, "Cost should be non-negative")
    h.assert_true(cost_2 >= 0, "Cost should be non-negative")

    # Tokens should accumulate
    h.assert_gt(tokens_2, tokens_1, "Tokens should accumulate across queries")

    # Summary should be consistent
    summary = h.tracer.summary()
    h.assert_eq(summary["total_tokens"], tokens_2, "Summary tokens should match accumulated total")


# ===== s09: Multi-Tool Selection =====
def s09_multi_tool(h: Harness) -> None:
    """Agent dynamically selects between different tools."""
    h.setup()

    # Query that should trigger web_search
    h.query("Search for the latest news about AI")
    tool_spans_1 = [s for s in h.tracer.get_spans() if s.kind.value == "TOOL"]
    search_used = any("web_search" in s.name for s in tool_spans_1)

    # Query that should trigger calculate
    h.query("Calculate 25 * 37 + 100")
    tool_spans_2 = [s for s in h.tracer.get_spans() if s.kind.value == "TOOL"]
    calc_used = any("calculate" in s.name for s in tool_spans_2)

    h.assert_true(search_used, "web_search should be selected for search queries")
    h.assert_true(calc_used, "calculate should be selected for math queries")

    # Should have used at least 2 different tools
    tool_names = set()
    for s in h.tracer.get_spans():
        if s.kind.value == "TOOL" and "tool_name" in s.attrs:
            tool_names.add(s.attrs["tool_name"])
    h.assert_gt(len(tool_names), 1, f"Should use >1 tool types, got: {tool_names}")


# ===== s10: Dashboard Generation =====
def s10_dashboard_gen(h: Harness) -> None:
    """Dashboard HTML is generated and contains all required sections."""
    h.setup()

    # Run a few queries to populate data
    h.query("Hello Prism!")
    h.query("Search for Python tutorials")
    h.query("Calculate 100 / 3")

    # Generate dashboard
    from prism.dashboard import generate_dashboard
    path = generate_dashboard(h.tracer, h.context, h.agent, output_dir="artifacts/test")

    h.assert_true(path.exists(), "Dashboard HTML file should be created")

    content = path.read_text(encoding="utf-8")
    h.assert_gt(len(content), 1000, "Dashboard should have substantial content")

    # Check for required sections
    h.assert_contains(content, "Span Tree", "Dashboard should have span tree section")
    h.assert_contains(content, "Flamegraph", "Dashboard should have flamegraph section")
    h.assert_contains(content, "Token Flow", "Dashboard should have token flow section")
    h.assert_contains(content, "Context Window", "Dashboard should have context window section")
    h.assert_contains(content, "Tool Usage", "Dashboard should have tool usage section")
    h.assert_contains(content, "Failure", "Dashboard should have failures section")
    h.assert_contains(content, "Conversation", "Dashboard should have conversation replay section")


# ===== Registry =====

SCENARIOS = [
    ("s01_basic_trace",       s01_basic_trace),
    ("s02_tool_trace",        s02_tool_trace),
    ("s03_nested_spans",      s03_nested_spans),
    ("s04_failure_recovery",  s04_failure_recovery),
    ("s05_long_conversation", s05_long_conversation),
    ("s06_context_budget",    s06_context_budget),
    ("s07_rolling_summary",   s07_rolling_summary),
    ("s08_cost_tracking",     s08_cost_tracking),
    ("s09_multi_tool",        s09_multi_tool),
    ("s10_dashboard_gen",     s10_dashboard_gen),
]
