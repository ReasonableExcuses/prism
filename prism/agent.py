"""
prism/agent.py — The hand-written plan → act → observe → repeat loop.

Thesis: this is a real agent, not a function-calling wrapper.  It plans
what to do, executes tools, observes results, and decides whether to
continue or respond.  There is no framework import.

The loop:
  1. PLAN  — LLM decides: answer directly or pick tool(s)
  2. ACT   — execute the chosen tool
  3. OBSERVE — did it work?  is the answer complete?
  4. REPEAT or RESPOND

Max 5 reasoning steps per query (configurable) to prevent infinite loops.
Every step is a traced span.  The final response includes citations —
which tools provided which facts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from prism.trace import Tracer, SpanKind
from prism.llm import LLMInterface, LLMResponse, ToolCall, TokenOverflowError
from prism.tools import ToolRegistry, ToolResult
from prism.context import ContextEngine


# ---------------------------------------------------------------------------
# Step model — records each reasoning step
# ---------------------------------------------------------------------------

@dataclass
class AgentStep:
    """One step in the agent's reasoning chain."""
    step_number: int
    action: str           # "plan" | "tool_call" | "observe" | "respond" | "error_recovery"
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[str] = None
    tool_success: Optional[bool] = None
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Agent response
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    """The agent's final response to a user query."""
    content: str
    steps: list[AgentStep]
    citations: list[dict[str, str]]   # [{tool, query, finding}]
    total_steps: int
    tokens_used: int
    cost_usd: float
    duration_ms: float


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """
    The core agent loop.

    Every query triggers:
      1. Context assembly (build_context)
      2. LLM planning (what tools to use)
      3. Tool execution (with error recovery)
      4. Observation and re-planning
      5. Final synthesis

    All inside traced spans.
    """

    def __init__(
        self,
        llm: LLMInterface,
        tool_registry: ToolRegistry,
        context_engine: ContextEngine,
        tracer: Tracer,
        max_steps: int = 5,
    ):
        self.llm = llm
        self.tools = tool_registry
        self.context = context_engine
        self.tracer = tracer
        self.max_steps = max_steps
        self._step_count = 0
        self._steps: list[AgentStep] = []
        self._citations: list[dict[str, str]] = []

    def run(self, user_input: str) -> AgentResponse:
        """
        Process a user query through the full agent loop.

        Returns an AgentResponse with the answer, reasoning steps,
        citations, and cost/token accounting.
        """
        start_time = time.time()
        self._steps = []
        self._citations = []
        self._step_count = 0

        with self.tracer.span("agent_run", SpanKind.AGENT) as root_span:
            root_span.set_attr("user_input", user_input[:200])

            # Record the user turn in context
            self.context.add_turn("user", user_input)

            # Build context with tracing
            messages = self.context.build_context(self.tracer)

            # Agent loop
            response_content = self._agent_loop(messages, user_input, root_span)

            # Record the assistant response in context
            self.context.add_turn("assistant", response_content)

            # Build the response
            duration_ms = (time.time() - start_time) * 1000
            root_span.set_attr("total_steps", self._step_count)
            root_span.set_attr("citations_count", len(self._citations))
            root_span.set_attr("response_length", len(response_content))

            return AgentResponse(
                content=response_content,
                steps=list(self._steps),
                citations=list(self._citations),
                total_steps=self._step_count,
                tokens_used=self.tracer.total_tokens,
                cost_usd=self.tracer.total_cost_usd,
                duration_ms=duration_ms,
            )

    def _agent_loop(
        self,
        messages: list[dict[str, str]],
        user_input: str,
        parent_span,
    ) -> str:
        """
        The core loop: plan → act → observe → repeat.
        Returns the final response content.
        """
        for step in range(self.max_steps):
            self._step_count += 1

            with self.tracer.span(f"step_{step + 1}", SpanKind.AGENT) as step_span:
                step_span.set_attr("step_number", step + 1)
                step_span.set_attr("max_steps", self.max_steps)

                # PLAN: ask the LLM what to do
                with self.tracer.span("plan", SpanKind.AGENT) as plan_span:
                    # Token overflow recovery: compress context and retry
                    llm_response = None
                    for compress_attempt in range(3):  # original + 2 retries
                        try:
                            llm_response = self.llm.run(
                                task="plan_and_act",
                                messages=messages,
                                tools=self.tools.schemas,
                                tracer=self.tracer,
                            )
                            break  # success
                        except TokenOverflowError as e:
                            if compress_attempt >= 2:
                                # Exhausted retries — fall back to a graceful message
                                plan_span.set_error(f"Token overflow after {compress_attempt + 1} compression attempts: {e}")
                                self._steps.append(AgentStep(
                                    step_number=self._step_count,
                                    action="error_recovery",
                                    content=f"Context too large even after compression. Error: {e}",
                                ))
                                return (
                                    "I'm sorry, but the conversation context has grown too large for the model to process. "
                                    "I've compressed the history as much as possible. Please try a shorter question, "
                                    "or reset the session in the sidebar for a fresh start."
                                )

                            # Log the compression recovery step
                            self._steps.append(AgentStep(
                                step_number=self._step_count,
                                action="error_recovery",
                                content=f"Token overflow detected (attempt {compress_attempt + 1}). Compressing context and retrying...",
                            ))
                            plan_span.add_event("token_overflow_recovery",
                                                attempt=compress_attempt + 1,
                                                error=str(e)[:200])

                            # Compress and rebuild
                            self.context.force_compress(self.tracer)
                            messages = self.context.build_context(self.tracer)

                    plan_span.set_attr("has_tool_calls", bool(llm_response.tool_calls))
                    plan_span.set_attr("has_content", bool(llm_response.content))

                # DECIDE: tool call or direct response?
                if llm_response.tool_calls:
                    # ACT: execute tool calls
                    for tc in llm_response.tool_calls:
                        tool_result = self._execute_tool(tc, step_span)

                        # Record step
                        self._steps.append(AgentStep(
                            step_number=self._step_count,
                            action="tool_call",
                            content=f"Called {tc.name}",
                            tool_name=tc.name,
                            tool_args=tc.arguments,
                            tool_result=tool_result.to_message()[:500],
                            tool_success=tool_result.success,
                        ))

                        # OBSERVE: add tool result to messages
                        result_msg = tool_result.to_message()
                        messages.append({
                            "role": "assistant",
                            "content": f"I'll use {tc.name} to help answer this."
                        })
                        messages.append({
                            "role": "user",
                            "content": f"[Tool Result from {tc.name}]:\n{result_msg}"
                        })

                        # Record the tool result in context
                        self.context.add_turn("tool", result_msg[:500], tool_name=tc.name)

                        # Track citation
                        if tool_result.success:
                            self._citations.append({
                                "tool": tc.name,
                                "query": json.dumps(tc.arguments)[:100],
                                "finding": str(tool_result.data)[:200] if tool_result.data else "",
                            })

                        # Error recovery
                        if not tool_result.success:
                            self._steps.append(AgentStep(
                                step_number=self._step_count,
                                action="error_recovery",
                                content=f"Tool {tc.name} failed: {tool_result.error}. Hint: {tool_result.hint}",
                            ))
                            step_span.add_event("error_recovery",
                                                tool=tc.name,
                                                error=tool_result.error or "",
                                                hint=tool_result.hint or "")

                    # Continue loop to synthesise or use another tool
                    continue

                elif llm_response.content:
                    # RESPOND: the LLM gave a direct answer
                    self._steps.append(AgentStep(
                        step_number=self._step_count,
                        action="respond",
                        content=llm_response.content[:200],
                    ))
                    step_span.set_attr("action", "direct_response")
                    return llm_response.content

                else:
                    # Edge case: LLM returned nothing
                    step_span.set_error("LLM returned empty response")
                    self._steps.append(AgentStep(
                        step_number=self._step_count,
                        action="error_recovery",
                        content="LLM returned empty response, retrying...",
                    ))
                    # Add a nudge
                    messages.append({
                        "role": "user",
                        "content": "Please provide a response based on the information gathered so far."
                    })
                    continue

        # Max steps reached — synthesise from what we have
        with self.tracer.span("final_synthesis", SpanKind.LLM) as synth_span:
            synth_span.add_event("max_steps_reached", max_steps=self.max_steps)
            messages.append({
                "role": "user",
                "content": "Please synthesize a final answer from everything you've gathered. Include citations for your sources."
            })
            final = self.llm.run(
                task="synthesize",
                messages=messages,
                tools=None,  # No tools for final synthesis
                tracer=self.tracer,
            )
            self._steps.append(AgentStep(
                step_number=self._step_count + 1,
                action="respond",
                content=final.content[:200],
            ))
            return final.content or "I wasn't able to find a complete answer. Please try rephrasing your question."

    def _execute_tool(self, tc: ToolCall, parent_span) -> ToolResult:
        """Execute a tool call with error handling."""
        parent_span.set_attr(f"tool_{tc.name}_args", tc.arguments)
        return self.tools.execute(tc.name, tc.arguments)

    # -----------------------------------------------------------------------
    # Inspection
    # -----------------------------------------------------------------------

    def get_steps(self) -> list[AgentStep]:
        """Return the reasoning steps from the last run."""
        return list(self._steps)

    def get_citations(self) -> list[dict[str, str]]:
        """Return citations from the last run."""
        return list(self._citations)
