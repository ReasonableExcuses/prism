"""
prism/trace.py — Span-level tracing for every operation in the agent.

Thesis: if it isn't traced, it didn't happen.  Every LLM call, tool call,
context mutation, and agent decision runs inside a Span.  The Tracer is
passed explicitly — no globals, no thread-locals — so the call graph is
visible in the type signatures.

Emits JSONL to artifacts/traces/<run_id>.jsonl.  Each line is one span.
Roll-ups (total_tokens, cost_usd, wall_ms, errors) are computed on close.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Generator


# ---------------------------------------------------------------------------
# Span kinds — colour-coded in the terminal tree
# ---------------------------------------------------------------------------

class SpanKind(str, Enum):
    AGENT    = "AGENT"
    LLM      = "LLM"
    TOOL     = "TOOL"
    CONTEXT  = "CONTEXT"
    RETRIEVE = "RETRIEVE"
    SYSTEM   = "SYSTEM"


# ---------------------------------------------------------------------------
# Pricing table — honest about what costs zero
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    # model_name -> {"input": $/1M tokens, "output": $/1M tokens}
    "deterministic":       {"input": 0.0,   "output": 0.0},
    "gemini-2.0-flash":    {"input": 0.10,  "output": 0.40},
    "gemini-2.5-flash":    {"input": 0.15,  "output": 0.60},
    "gemini-2.5-pro":      {"input": 1.25,  "output": 10.0},
    "gpt-4o":              {"input": 2.50,  "output": 10.0},
    "gpt-4o-mini":         {"input": 0.15,  "output": 0.60},
    "claude-sonnet-4":     {"input": 3.00,  "output": 15.0},
    "claude-haiku-3.5":    {"input": 0.80,  "output": 4.00},
}


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost from the pricing table."""
    prices = PRICING.get(model, PRICING["deterministic"])
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Event — lightweight annotation inside a span
# ---------------------------------------------------------------------------

@dataclass
class SpanEvent:
    name: str
    timestamp: float = field(default_factory=time.time)
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "timestamp": self.timestamp, "attrs": self.attrs}


# ---------------------------------------------------------------------------
# Span — the atom of observability
# ---------------------------------------------------------------------------

@dataclass
class Span:
    """A single unit of traced work."""

    span_id: str
    name: str
    kind: SpanKind
    parent_id: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    model: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    attrs: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    error: Optional[str] = None
    children: list[Span] = field(default_factory=list)

    # -- mutable state helpers ------------------------------------------------

    def set_model(self, model: str) -> None:
        self.model = model

    def set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        if self.model:
            self.cost_usd = _cost(self.model, input_tokens, output_tokens)

    def set_attr(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def set_error(self, error: str) -> None:
        self.error = error

    def add_event(self, name: str, **attrs: Any) -> None:
        self.events.append(SpanEvent(name=name, attrs=attrs))

    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "span_id": self.span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration_ms, 2),
        }
        if self.parent_id:
            d["parent_id"] = self.parent_id
        if self.model:
            d["model"] = self.model
        if self.input_tokens or self.output_tokens:
            d["input_tokens"] = self.input_tokens
            d["output_tokens"] = self.output_tokens
            d["cost_usd"] = round(self.cost_usd, 8)
        if self.attrs:
            d["attrs"] = self.attrs
        if self.events:
            d["events"] = [e.to_dict() for e in self.events]
        if self.error:
            d["error"] = self.error
        return d

    def tree_dict(self) -> dict:
        """Recursive tree representation for the dashboard."""
        d = self.to_dict()
        if self.children:
            d["children"] = [c.tree_dict() for c in self.children]
        return d


# ---------------------------------------------------------------------------
# Tracer — one per agent run
# ---------------------------------------------------------------------------

class Tracer:
    """
    Creates and manages a tree of Spans for a single agent run.

    Usage::

        tracer = Tracer()
        with tracer.span("query", SpanKind.AGENT) as s:
            with tracer.span("plan", SpanKind.LLM) as llm_s:
                llm_s.set_tokens(100, 50)
            with tracer.span("search", SpanKind.TOOL) as tool_s:
                tool_s.set_attr("query", "python async")
        tracer.save()
    """

    def __init__(self, artifacts_dir: str = "artifacts", run_id: Optional[str] = None):
        self.run_id: str = run_id or uuid.uuid4().hex[:12]
        self.started_at: str = datetime.now(timezone.utc).isoformat()
        self._artifacts_dir = Path(artifacts_dir)
        self._root_spans: list[Span] = []
        self._all_spans: list[Span] = []
        self._span_stack: list[Span] = []
        self._closed = False

    # -- span creation --------------------------------------------------------

    @contextmanager
    def span(self, name: str, kind: SpanKind | str = SpanKind.SYSTEM) -> Generator[Span, None, None]:
        """Open a new span as a child of the current one."""
        if isinstance(kind, str):
            kind = SpanKind(kind)

        parent = self._span_stack[-1] if self._span_stack else None
        s = Span(
            span_id=uuid.uuid4().hex[:10],
            name=name,
            kind=kind,
            parent_id=parent.span_id if parent else None,
            start_time=time.time(),
        )
        if parent:
            parent.children.append(s)
        else:
            self._root_spans.append(s)

        self._all_spans.append(s)
        self._span_stack.append(s)
        try:
            yield s
        except Exception as exc:
            s.set_error(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            s.end_time = time.time()
            self._span_stack.pop()

    # -- roll-ups -------------------------------------------------------------

    @property
    def total_llm_calls(self) -> int:
        return sum(1 for s in self._all_spans if s.kind == SpanKind.LLM)

    @property
    def total_tool_calls(self) -> int:
        return sum(1 for s in self._all_spans if s.kind == SpanKind.TOOL)

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens for s in self._all_spans)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens for s in self._all_spans)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self._all_spans)

    @property
    def total_errors(self) -> int:
        return sum(1 for s in self._all_spans if s.error)

    @property
    def wall_ms(self) -> float:
        if not self._root_spans:
            return 0.0
        start = min(s.start_time for s in self._root_spans)
        end = max(s.end_time for s in self._root_spans if s.end_time)
        return (end - start) * 1000 if end > start else 0.0

    def get_spans(self) -> list[Span]:
        return list(self._all_spans)

    def get_root_spans(self) -> list[Span]:
        return list(self._root_spans)

    # -- summary dict (for dashboard) -----------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "llm_calls": self.total_llm_calls,
            "tool_calls": self.total_tool_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.total_cost_usd, 6),
            "wall_ms": round(self.wall_ms, 1),
            "errors": self.total_errors,
            "span_count": len(self._all_spans),
        }

    # -- persistence -----------------------------------------------------------

    def save(self) -> Path:
        """Write all spans as JSONL to artifacts/traces/<run_id>.jsonl."""
        traces_dir = self._artifacts_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        path = traces_dir / f"{self.run_id}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            # First line: run metadata
            meta = {
                "type": "run_meta",
                "run_id": self.run_id,
                "started_at": self.started_at,
                **self.summary(),
            }
            f.write(json.dumps(meta) + "\n")
            # Remaining lines: one per span
            for span in self._all_spans:
                f.write(json.dumps(span.to_dict()) + "\n")
        self._closed = True
        return path

    def get_tree_data(self) -> list[dict]:
        """Return the full span tree for dashboard rendering."""
        return [s.tree_dict() for s in self._root_spans]

    # -- terminal rendering ---------------------------------------------------

    _KIND_COLORS = {
        SpanKind.AGENT:    "\033[1;35m",   # bold magenta
        SpanKind.LLM:      "\033[1;36m",   # bold cyan
        SpanKind.TOOL:     "\033[1;33m",   # bold yellow
        SpanKind.CONTEXT:  "\033[1;32m",   # bold green
        SpanKind.RETRIEVE: "\033[1;34m",   # bold blue
        SpanKind.SYSTEM:   "\033[0;37m",   # grey
    }
    _RESET = "\033[0m"
    _DIM = "\033[2m"
    _RED = "\033[1;31m"

    def print_tree(self) -> str:
        """Render the span tree to a string (and print it)."""
        lines: list[str] = []
        lines.append(f"\n{'═' * 60}")
        lines.append(f"  🔮 PRISM TRACE — Run {self.run_id}")
        lines.append(f"{'═' * 60}")

        def _render(span: Span, depth: int = 0) -> None:
            indent = "  │ " * depth
            color = self._KIND_COLORS.get(span.kind, self._RESET)
            icon = {
                SpanKind.AGENT:    "🤖",
                SpanKind.LLM:      "🧠",
                SpanKind.TOOL:     "🔧",
                SpanKind.CONTEXT:  "📦",
                SpanKind.RETRIEVE: "🔍",
                SpanKind.SYSTEM:   "⚙️",
            }.get(span.kind, "•")

            dur = f"{span.duration_ms:.0f}ms"
            tokens = ""
            if span.input_tokens or span.output_tokens:
                tokens = f" [{span.input_tokens}→{span.output_tokens} tok]"
            cost = ""
            if span.cost_usd > 0:
                cost = f" ${span.cost_usd:.6f}"
            err = ""
            if span.error:
                err = f" {self._RED}✗ {span.error}{self._RESET}"

            line = (
                f"{indent}{icon} {color}{span.name}{self._RESET}"
                f" {self._DIM}{dur}{tokens}{cost}{self._RESET}{err}"
            )
            lines.append(line)

            for event in span.events:
                eline = f"{indent}  │ 📌 {self._DIM}{event.name}"
                if event.attrs:
                    eline += f" {event.attrs}"
                eline += self._RESET
                lines.append(eline)

            for child in span.children:
                _render(child, depth + 1)

        for root in self._root_spans:
            _render(root)

        # Footer
        s = self.summary()
        lines.append(f"{'─' * 60}")
        lines.append(
            f"  LLM calls: {s['llm_calls']}  |  Tool calls: {s['tool_calls']}  |  "
            f"Tokens: {s['total_tokens']}  |  Cost: ${s['cost_usd']:.6f}  |  "
            f"Time: {s['wall_ms']:.0f}ms  |  Errors: {s['errors']}"
        )
        lines.append(f"{'═' * 60}\n")

        output = "\n".join(lines)
        print(output)
        return output
