"""
prism/context.py — Context engineering for long conversations.

Thesis: the model should see exactly what it needs and nothing more.
Every piece of context competes for a finite token budget.  When the
budget is exceeded, we summarise, truncate, or drop — and every such
decision is traced so a judge can see exactly what the model saw.

Strategy (Hybrid 3-tier):
  1. Sliding Window  — last N turns verbatim
  2. Rolling Summary — everything older compressed into a running summary
  3. Key Facts       — critical preferences / constraints, never summarised away

Token Budget:
  - System prompt:  500 tokens
  - Key facts:      200 tokens
  - Rolling summary: 400 tokens
  - Recent turns:   800 tokens
  - Tool results:   600 tokens
  - Response room: 1000 tokens
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from prism.trace import Tracer, SpanKind


# ---------------------------------------------------------------------------
# Budget configuration
# ---------------------------------------------------------------------------

@dataclass
class ContextBudget:
    """Token budget allocation for different context sections."""
    system_prompt: int = 500
    key_facts: int = 200
    rolling_summary: int = 400
    recent_turns: int = 800
    tool_results: int = 600
    response_room: int = 1000

    @property
    def total(self) -> int:
        return (self.system_prompt + self.key_facts + self.rolling_summary +
                self.recent_turns + self.tool_results + self.response_room)


# ---------------------------------------------------------------------------
# Turn model
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """One conversational turn."""
    role: str           # "user" | "assistant" | "tool"
    content: str
    turn_number: int
    timestamp: float = field(default_factory=time.time)
    tool_name: Optional[str] = None
    tokens: int = 0     # estimated

    def to_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# Context snapshot — what the model actually sees
# ---------------------------------------------------------------------------

@dataclass
class ContextSnapshot:
    """Records exactly what was packed into the model's context."""
    system_prompt: str
    key_facts: list[str]
    rolling_summary: str
    recent_turns: list[Turn]
    dropped_turns: int          # how many turns were summarised
    total_turns: int            # total conversation length
    budget: ContextBudget
    allocation: dict[str, int]  # actual tokens used per section
    utilization_pct: float      # % of total budget used


# ---------------------------------------------------------------------------
# Context Engine
# ---------------------------------------------------------------------------

class ContextEngine:
    """
    Manages the conversation context with a 3-tier strategy:
      1. Sliding window of recent turns
      2. Rolling summary of older turns
      3. Key facts that persist forever
    """

    SYSTEM_PROMPT = """You are Prism, an AI research assistant with full observability.

You have access to tools: web_search, read_url, calculate, analyze_data, take_note.

RULES:
- When asked a factual question, use web_search or read_url to find the answer. Do NOT make up facts.
- When asked to do math, use the calculate tool. Do NOT compute in your head.
- When you find important information, use take_note to save it.
- Always cite which tool/source gave you the answer.
- If a tool fails, explain what happened and try an alternative approach.
- Be concise but thorough. Structure your answers with clear formatting.

IMPORTANT: You MUST use tools for research. Do not answer from memory alone."""

    def __init__(
        self,
        budget: ContextBudget | None = None,
        window_size: int = 6,
        tracer: Tracer | None = None,
    ):
        self.budget = budget or ContextBudget()
        self.window_size = window_size
        self._tracer = tracer

        # State
        self._turns: list[Turn] = []
        self._key_facts: list[str] = []
        self._rolling_summary: str = ""
        self._turn_counter: int = 0
        self._snapshots: list[ContextSnapshot] = []  # history of context allocations

    @property
    def turn_count(self) -> int:
        return self._turn_counter

    @property
    def snapshots(self) -> list[ContextSnapshot]:
        return list(self._snapshots)

    # -----------------------------------------------------------------------
    # Adding turns
    # -----------------------------------------------------------------------

    def add_turn(self, role: str, content: str, tool_name: str | None = None) -> None:
        """Record a new conversational turn."""
        self._turn_counter += 1
        turn = Turn(
            role=role,
            content=content,
            turn_number=self._turn_counter,
            tool_name=tool_name,
            tokens=self._estimate_tokens(content),
        )
        self._turns.append(turn)

    def add_key_fact(self, fact: str) -> None:
        """Add a fact that should persist regardless of summarisation."""
        if fact not in self._key_facts:
            self._key_facts.append(fact)

    # -----------------------------------------------------------------------
    # Building context — the main event
    # -----------------------------------------------------------------------

    def build_context(self, tracer: Tracer | None = None) -> list[dict[str, str]]:
        """
        Build the message list for the LLM, respecting token budgets.

        Returns a list of messages suitable for any LLM provider.
        Every decision (what was kept, what was dropped, why) is traced.
        """
        t = tracer or self._tracer
        if not t:
            return self._build_context_untraced()

        with t.span("context_engineering", SpanKind.CONTEXT) as ctx_span:
            messages = self._build_context_inner(ctx_span, t)

            # Record the snapshot
            snapshot = self._snapshots[-1] if self._snapshots else None
            if snapshot:
                ctx_span.set_attr("total_turns", snapshot.total_turns)
                ctx_span.set_attr("recent_turns", len(snapshot.recent_turns))
                ctx_span.set_attr("dropped_turns", snapshot.dropped_turns)
                ctx_span.set_attr("has_summary", bool(snapshot.rolling_summary))
                ctx_span.set_attr("key_facts_count", len(snapshot.key_facts))
                ctx_span.set_attr("utilization_pct", round(snapshot.utilization_pct, 1))
                ctx_span.set_attr("allocation", snapshot.allocation)

            return messages

    def _build_context_untraced(self) -> list[dict[str, str]]:
        """Build context without tracing (for testing)."""
        messages: list[dict[str, str]] = []

        # 1. System prompt
        messages.append({"role": "system", "content": self.SYSTEM_PROMPT})

        # 2. Key facts
        if self._key_facts:
            facts_text = "KEY FACTS (always relevant):\n" + "\n".join(f"• {f}" for f in self._key_facts)
            messages.append({"role": "system", "content": facts_text})

        # 3. Rolling summary
        if self._rolling_summary:
            messages.append({"role": "system", "content": f"CONVERSATION SUMMARY:\n{self._rolling_summary}"})

        # 4. Recent turns
        recent = self._turns[-self.window_size:] if len(self._turns) > self.window_size else self._turns
        for turn in recent:
            messages.append(turn.to_message())

        return messages

    def _build_context_inner(self, ctx_span, tracer: Tracer) -> list[dict[str, str]]:
        """Build context with full tracing of every decision."""
        messages: list[dict[str, str]] = []
        allocation: dict[str, int] = {}

        # 1. System prompt (always included)
        sys_tokens = self._estimate_tokens(self.SYSTEM_PROMPT)
        messages.append({"role": "system", "content": self.SYSTEM_PROMPT})
        allocation["system_prompt"] = sys_tokens

        # 2. Key facts
        facts_tokens = 0
        if self._key_facts:
            with tracer.span("pack_key_facts", SpanKind.CONTEXT) as fs:
                facts_text = "KEY FACTS (always relevant):\n" + "\n".join(f"• {f}" for f in self._key_facts)
                facts_tokens = self._estimate_tokens(facts_text)

                # Truncate if over budget
                if facts_tokens > self.budget.key_facts:
                    # Keep the most recent facts
                    kept_facts = []
                    running = 0
                    for fact in reversed(self._key_facts):
                        ft = self._estimate_tokens(fact) + 5  # overhead
                        if running + ft <= self.budget.key_facts:
                            kept_facts.insert(0, fact)
                            running += ft
                        else:
                            fs.add_event("fact_dropped", fact=fact, reason="budget")

                    facts_text = "KEY FACTS (always relevant):\n" + "\n".join(f"• {f}" for f in kept_facts)
                    facts_tokens = self._estimate_tokens(facts_text)

                messages.append({"role": "system", "content": facts_text})
                fs.set_attr("facts_count", len(self._key_facts))
                fs.set_attr("tokens", facts_tokens)
        allocation["key_facts"] = facts_tokens

        # 3. Rolling summary of older turns
        summary_tokens = 0
        if self._rolling_summary:
            with tracer.span("pack_summary", SpanKind.CONTEXT) as ss:
                summary_text = f"CONVERSATION SUMMARY:\n{self._rolling_summary}"
                summary_tokens = self._estimate_tokens(summary_text)

                # Truncate if needed
                if summary_tokens > self.budget.rolling_summary:
                    max_chars = self.budget.rolling_summary * 4  # rough
                    summary_text = summary_text[:max_chars] + "... [truncated]"
                    summary_tokens = self._estimate_tokens(summary_text)
                    ss.add_event("summary_truncated",
                                 original_tokens=self._estimate_tokens(self._rolling_summary),
                                 kept_tokens=summary_tokens)

                messages.append({"role": "system", "content": summary_text})
                ss.set_attr("tokens", summary_tokens)
        allocation["rolling_summary"] = summary_tokens

        # 4. Recent turns (sliding window)
        with tracer.span("pack_recent_turns", SpanKind.CONTEXT) as ts:
            recent = self._turns[-self.window_size:] if len(self._turns) > self.window_size else self._turns[:]
            dropped = max(0, len(self._turns) - self.window_size)

            # Check token budget
            turn_tokens = sum(t.tokens for t in recent)
            while turn_tokens > self.budget.recent_turns and len(recent) > 2:
                dropped_turn = recent.pop(0)
                dropped += 1
                turn_tokens -= dropped_turn.tokens
                ts.add_event("turn_dropped",
                             turn_number=dropped_turn.turn_number,
                             role=dropped_turn.role,
                             tokens=dropped_turn.tokens,
                             reason="budget")

            for turn in recent:
                messages.append(turn.to_message())

            ts.set_attr("kept_turns", len(recent))
            ts.set_attr("dropped_turns", dropped)
            ts.set_attr("tokens", turn_tokens)
            allocation["recent_turns"] = turn_tokens

        # Update rolling summary if we dropped turns
        if dropped > 0 and len(self._turns) > self.window_size:
            self._update_summary(self._turns[:dropped], tracer)

        # Calculate utilization
        total_used = sum(allocation.values())
        total_avail = self.budget.total - self.budget.response_room
        utilization = (total_used / total_avail * 100) if total_avail > 0 else 0

        # Record snapshot
        snapshot = ContextSnapshot(
            system_prompt=self.SYSTEM_PROMPT,
            key_facts=list(self._key_facts),
            rolling_summary=self._rolling_summary,
            recent_turns=recent,
            dropped_turns=dropped,
            total_turns=self._turn_counter,
            budget=self.budget,
            allocation=allocation,
            utilization_pct=utilization,
        )
        self._snapshots.append(snapshot)

        return messages

    # -----------------------------------------------------------------------
    # Rolling summarisation
    # -----------------------------------------------------------------------

    def _update_summary(self, old_turns: list[Turn], tracer: Tracer) -> None:
        """
        Compress older turns into the rolling summary.
        This is a deterministic summary (template-based) under DeterministicLLM.
        """
        with tracer.span("summarize_old_turns", SpanKind.CONTEXT) as span:
            # Build a summary from the old turns
            user_messages = [t.content for t in old_turns if t.role == "user"]
            assistant_messages = [t.content for t in old_turns if t.role == "assistant"]

            new_summary_parts = []
            if user_messages:
                topics = "; ".join(msg[:60] for msg in user_messages[-5:])
                new_summary_parts.append(f"User discussed: {topics}")
            if assistant_messages:
                points = "; ".join(msg[:60] for msg in assistant_messages[-3:])
                new_summary_parts.append(f"Assistant covered: {points}")

            new_section = ". ".join(new_summary_parts)

            # Merge with existing summary
            if self._rolling_summary:
                self._rolling_summary = f"{self._rolling_summary}\n{new_section}"
            else:
                self._rolling_summary = new_section

            # Trim if too long
            max_chars = self.budget.rolling_summary * 4
            if len(self._rolling_summary) > max_chars:
                self._rolling_summary = self._rolling_summary[-max_chars:]
                span.add_event("summary_trimmed", max_chars=max_chars)

            span.set_attr("turns_summarized", len(old_turns))
            span.set_attr("summary_length", len(self._rolling_summary))
            span.set_attr("summary_preview", self._rolling_summary[:200])

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(text) // 4)

    def get_allocation_history(self) -> list[dict]:
        """Return the allocation history for the dashboard."""
        history = []
        for i, snap in enumerate(self._snapshots):
            history.append({
                "turn": i + 1,
                "total_turns": snap.total_turns,
                "allocation": snap.allocation,
                "utilization_pct": round(snap.utilization_pct, 1),
                "dropped_turns": snap.dropped_turns,
                "key_facts_count": len(snap.key_facts),
                "has_summary": bool(snap.rolling_summary),
            })
        return history

    def get_status(self) -> dict:
        """Return current context engine status."""
        return {
            "total_turns": self._turn_counter,
            "stored_turns": len(self._turns),
            "window_size": self.window_size,
            "key_facts": len(self._key_facts),
            "has_summary": bool(self._rolling_summary),
            "summary_length": len(self._rolling_summary),
            "budget": {
                "system_prompt": self.budget.system_prompt,
                "key_facts": self.budget.key_facts,
                "rolling_summary": self.budget.rolling_summary,
                "recent_turns": self.budget.recent_turns,
                "tool_results": self.budget.tool_results,
                "response_room": self.budget.response_room,
                "total": self.budget.total,
            },
        }

    def reset(self) -> None:
        """Clear all state."""
        self._turns.clear()
        self._key_facts.clear()
        self._rolling_summary = ""
        self._turn_counter = 0
        self._snapshots.clear()
