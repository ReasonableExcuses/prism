"""
prism/llm.py — One interface, multiple implementations.

Thesis: the agent logic must be identical regardless of which model runs
underneath.  ``LLM.run(task, messages, tools, tracer)`` is the only
call site; everything above it is model-agnostic.

Implementations:
  - DeterministicLLM  — rule-based stub, no API key, reproducible evals
  - GeminiLLM         — Google Gemini
  - OpenAILLM         — OpenAI GPT
  - AnthropicLLM      — Anthropic Claude

The deterministic stub emits the exact JSON schema the real prompts
request, so the eval suite measures the agent, not the model's mood.
"""

from __future__ import annotations

import json
import os
import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from prism.trace import Tracer, SpanKind


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = "unknown"
    raw: Any = None  # provider-specific raw response


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class LLMInterface(ABC):
    """Every implementation must satisfy this contract."""

    model_name: str = "unknown"

    @abstractmethod
    def run(
        self,
        task: str,
        messages: list[dict[str, str]],
        tools: list[dict] | None,
        tracer: Tracer,
    ) -> LLMResponse:
        """Send messages to the model. Returns a unified response."""
        ...

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# DeterministicLLM — the reproducible stub
# ---------------------------------------------------------------------------

class DeterministicLLM(LLMInterface):
    """
    A rule-based stand-in that emits structured responses matching the
    exact JSON schema the real prompts request.  This is why the eval
    suite is reproducible to the byte: it measures the agent, not the
    model.

    Pattern matching rules:
      - If tools are available and the query looks like it needs one,
        emit a tool_call.
      - If the conversation contains tool results, synthesise an answer.
      - Otherwise, respond conversationally.
    """

    model_name = "deterministic"

    # Keywords that trigger specific tool selections
    _TOOL_TRIGGERS: dict[str, list[str]] = {
        "web_search":    ["search", "find", "look up", "latest", "news", "what is", "who is", "current"],
        "read_url":      ["read", "fetch", "url", "http", "website", "page", "article"],
        "calculate":     ["calculate", "compute", "math", "sum", "average", "multiply", "divide", "add", "subtract", "percent", "formula"],
        "analyze_data":  ["analyze", "data", "csv", "json", "statistics", "stats", "chart", "table", "dataset"],
        "take_note":     ["note", "remember", "save", "record", "jot"],
    }

    def run(
        self,
        task: str,
        messages: list[dict[str, str]],
        tools: list[dict] | None,
        tracer: Tracer,
    ) -> LLMResponse:
        with tracer.span(f"llm:{task}", SpanKind.LLM) as span:
            span.set_model(self.model_name)

            last_real_user = ""
            last_real_user_idx = -1
            tool_results_content = []

            # Find the last real user message (not a tool result)
            for i, msg in enumerate(messages):
                content = msg.get("content", "")
                if msg.get("role") == "user" and not content.startswith("[Tool Result"):
                    last_real_user = content
                    last_real_user_idx = i

            # Collect tool results that appear AFTER the last real user message
            for i, msg in enumerate(messages):
                if i > last_real_user_idx:
                    content = msg.get("content", "")
                    if msg.get("role") == "tool" or (msg.get("role") == "user" and content.startswith("[Tool Result")):
                        tool_results_content.append(content)

            has_tool_results = len(tool_results_content) > 0
            query_lower = last_real_user.lower()
            input_text = json.dumps(messages)
            input_tokens = self._estimate_tokens(input_text)

            # If we have tool results for THIS query, synthesise an answer
            if has_tool_results:
                response = self._synthesise_answer(query_lower, tool_results_content, messages)
                output_tokens = self._estimate_tokens(response.content)
                response.input_tokens = input_tokens
                response.output_tokens = output_tokens
                response.model = self.model_name
                span.set_tokens(input_tokens, output_tokens)
                span.set_attr("task", task)
                span.set_attr("has_tool_results", True)
                return response

            # If tools available, try to match a tool
            if tools:
                tool_names = [t["name"] for t in tools]
                matched = self._match_tool(query_lower, tool_names)
                if matched:
                    tool_call = self._build_tool_call(matched, query_lower, messages)
                    response = LLMResponse(
                        content="",
                        tool_calls=[tool_call],
                        model=self.model_name,
                    )
                    output_tokens = self._estimate_tokens(json.dumps(tool_call.arguments))
                    response.input_tokens = input_tokens
                    response.output_tokens = output_tokens
                    span.set_tokens(input_tokens, output_tokens)
                    span.set_attr("task", task)
                    span.set_attr("tool_selected", matched)
                    return response

            # Fallback: conversational response
            content = self._conversational_response(query_lower, messages)
            output_tokens = self._estimate_tokens(content)
            span.set_tokens(input_tokens, output_tokens)
            span.set_attr("task", task)

            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.model_name,
            )

    def _match_tool(self, query: str, available: list[str]) -> Optional[str]:
        """Pick the best tool based on keyword matching."""
        scores: dict[str, int] = {}
        for tool_name, triggers in self._TOOL_TRIGGERS.items():
            if tool_name in available:
                score = sum(1 for t in triggers if t in query)
                if score > 0:
                    scores[tool_name] = score

        # Also detect math expressions for calculate
        if "calculate" in available:
            import re
            # Check for math operators with numbers
            has_math = bool(re.search(r'\d+\s*[+\-*/]\s*\d+', query))
            if has_math:
                scores["calculate"] = scores.get("calculate", 0) + 5  # strong signal

        if scores:
            return max(scores, key=scores.get)
        return None

    def _build_tool_call(self, tool_name: str, query: str, messages: list[dict]) -> ToolCall:
        """Build a plausible tool call from the query."""
        # Extract the core search/action query
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break

        if tool_name == "web_search":
            return ToolCall(name="web_search", arguments={"query": last_user})
        elif tool_name == "read_url":
            # Try to find a URL in the message
            url_match = re.search(r'https?://\S+', last_user)
            url = url_match.group(0) if url_match else "https://example.com"
            return ToolCall(name="read_url", arguments={"url": url})
        elif tool_name == "calculate":
            # Extract mathematical expression
            expr = re.sub(r'[^0-9+\-*/().%\s]', '', last_user).strip()
            if not expr:
                expr = "1 + 1"
            return ToolCall(name="calculate", arguments={"expression": expr})
        elif tool_name == "analyze_data":
            return ToolCall(name="analyze_data", arguments={
                "data": last_user,
                "operation": "summarize"
            })
        elif tool_name == "take_note":
            return ToolCall(name="take_note", arguments={
                "content": last_user,
                "tag": "general"
            })
        return ToolCall(name=tool_name, arguments={"input": last_user})

    def _synthesise_answer(
        self,
        query: str,
        tool_results: list[str],
        messages: list[dict]
    ) -> LLMResponse:
        """Build an answer from tool results."""
        # Find the original user question
        original_question = ""
        for msg in messages:
            if msg.get("role") == "user":
                original_question = msg.get("content", "")

        combined = "\n\n".join(tool_results)
        # Truncate if too long
        if len(combined) > 2000:
            combined = combined[:2000] + "..."

        content = (
            f"Based on my research, here's what I found:\n\n"
            f"{combined}\n\n"
            f"This answers your question about: {original_question}"
        )
        return LLMResponse(content=content)

    def _conversational_response(self, query: str, messages: list[dict]) -> str:
        """Generate a simple conversational response."""
        if "hello" in query or "hi" in query:
            return "Hello! I'm Prism, your AI research assistant. I can search the web, read articles, do calculations, analyze data, and take notes. How can I help you?"
        if "help" in query:
            return ("I can help you with:\n"
                    "• **Web Search** — Find information online\n"
                    "• **Read URLs** — Extract content from web pages\n"
                    "• **Calculate** — Do math computations\n"
                    "• **Analyze Data** — Process and summarize data\n"
                    "• **Take Notes** — Save important findings\n\n"
                    "Just ask me a question!")
        if "thank" in query:
            return "You're welcome! Let me know if you need anything else."
        return (
            f"I understand you're asking about: \"{query[:100]}\".\n"
            "Let me help you with that. Could you be more specific about what you'd like me to research?"
        )


# ---------------------------------------------------------------------------
# GeminiLLM — Google Gemini
# ---------------------------------------------------------------------------

class GeminiLLM(LLMInterface):
    """Google Gemini implementation via direct REST API with auto-model resolution."""

    model_name = "gemini-3.6-flash"

    # Map deprecated, sunset, or preview model names to active supported ones
    MODEL_ALIASES = {
        "gemini-2.5-flash": "gemini-3.6-flash",
        "gemini-1.5-flash": "gemini-3.6-flash",
        "gemini-1.5-pro": "gemini-3.6-flash",
        "gemini-pro": "gemini-3.6-flash",
        "gemini-flash": "gemini-3.6-flash",
    }

    def __init__(self, model: str = "gemini-3.6-flash", api_key: str | None = None):
        mapped = self.MODEL_ALIASES.get(model, model)
        self.model_name = mapped
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set")

    def run(
        self,
        task: str,
        messages: list[dict[str, str]],
        tools: list[dict] | None,
        tracer: Tracer,
    ) -> LLMResponse:
        import requests

        with tracer.span(f"llm:{task}", SpanKind.LLM) as span:
            span.set_model(self.model_name)

            # Convert messages to Gemini format
            gemini_messages = []
            system_text = ""
            for msg in messages:
                role = msg["role"]
                content = msg.get("content", "")
                if role == "system":
                    system_text = content
                elif role == "user":
                    gemini_messages.append({"role": "user", "parts": [{"text": content}]})
                elif role == "assistant":
                    gemini_messages.append({"role": "model", "parts": [{"text": content}]})
                elif role == "tool":
                    gemini_messages.append({"role": "user", "parts": [{"text": f"[Tool Result]: {content}"}]})

            if not gemini_messages:
                gemini_messages = [{"role": "user", "parts": [{"text": "Hello"}]}]

            # Merge consecutive turns with the same role (required by Gemini API)
            merged_messages = []
            for m in gemini_messages:
                if merged_messages and merged_messages[-1]["role"] == m["role"]:
                    merged_messages[-1]["parts"].extend(m["parts"])
                else:
                    merged_messages.append(m)

            # Build tools schema
            gemini_tools = None
            if tools:
                declarations = []
                for t in tools:
                    params = t.get("parameters", {})
                    declarations.append({
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": {
                            "type": "OBJECT",
                            "properties": {
                                k: {
                                    "type": "STRING",
                                    "description": v.get("description", ""),
                                }
                                for k, v in params.get("properties", {}).items()
                            },
                            "required": params.get("required", []),
                        },
                    })
                gemini_tools = [{"function_declarations": declarations}]

            payload: dict[str, Any] = {"contents": merged_messages}
            if system_text:
                payload["system_instruction"] = {"parts": [{"text": system_text}]}
            if gemini_tools:
                payload["tools"] = gemini_tools

            # Try requested model with fallback to gemini-3.6-flash if 404
            models_to_try = [self.model_name]
            if self.model_name != "gemini-3.6-flash":
                models_to_try.append("gemini-3.6-flash")

            last_err = None
            for candidate_model in models_to_try:
                endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{candidate_model}:generateContent?key={self._api_key}"
                try:
                    resp = requests.post(endpoint, json=payload, timeout=50)
                    if resp.status_code == 404:
                        last_err = f"Model {candidate_model} returned 404 (not found or deprecated)"
                        continue
                    if resp.status_code != 200:
                        err_text = resp.text[:300]
                        try:
                            err_data = resp.json()
                            err_text = err_data.get("error", {}).get("message", err_text)
                        except Exception:
                            pass
                        raise RuntimeError(f"Gemini API Error ({resp.status_code}): {err_text}")

                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise RuntimeError("Gemini returned no candidates in response")

                    candidate = candidates[0]
                    content_obj = candidate.get("content", {})
                    parts = content_obj.get("parts", [])

                    content = ""
                    tool_calls = []
                    for part in parts:
                        if "text" in part and part["text"]:
                            content += part["text"]
                        if "functionCall" in part and part["functionCall"]:
                            fc = part["functionCall"]
                            tool_calls.append(ToolCall(
                                name=fc.get("name", ""),
                                arguments=dict(fc.get("args", {})),
                            ))

                    usage = data.get("usageMetadata", {})
                    input_tokens = usage.get("promptTokenCount", self._estimate_tokens(json.dumps(messages)))
                    output_tokens = usage.get("candidatesTokenCount", self._estimate_tokens(content))

                    span.set_tokens(input_tokens, output_tokens)
                    span.set_attr("task", task)
                    span.set_model(candidate_model)

                    return LLMResponse(
                        content=content,
                        tool_calls=tool_calls,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        model=candidate_model,
                        raw=data,
                    )
                except requests.exceptions.ReadTimeout:
                    timeout_msg = (
                        f"Google Gemini ReadTimeout (>50s on {candidate_model}). "
                        "Google's free-tier endpoint is experiencing severe regional latency or traffic queuing. "
                        "Recommendation: Switch to Groq (Free & Ultra Fast Llama 3.3 70B @ 500+ tok/s) or Deterministic mode in the sidebar."
                    )
                    span.set_error(timeout_msg)
                    raise RuntimeError(timeout_msg)
                except Exception as e:
                    last_err = str(e)
                    if "404" not in str(e):
                        span.set_error(str(e))
                        raise

            span.set_error(last_err or "Gemini call failed")
            raise RuntimeError(last_err or "Gemini call failed")


# ---------------------------------------------------------------------------
# OpenAICompatibleLLM — Generic REST client (OpenAI, Groq, OpenRouter, etc.)
# ---------------------------------------------------------------------------

class OpenAICompatibleLLM(LLMInterface):
    """
    Direct REST client for any OpenAI-compatible chat completion endpoint.
    Zero dependency beyond requests.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        provider_name: str = "OpenAI",
        env_key_name: str = "OPENAI_API_KEY",
        extra_headers: dict[str, str] | None = None,
    ):
        self.model_name = model
        self.base_url = base_url.rstrip("/")
        self.provider_name = provider_name
        self.extra_headers = extra_headers or {}
        self._api_key = api_key or os.environ.get(env_key_name) or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(f"{env_key_name} must be set")

    def run(
        self,
        task: str,
        messages: list[dict[str, str]],
        tools: list[dict] | None,
        tracer: Tracer,
    ) -> LLMResponse:
        import requests

        with tracer.span(f"llm:{task}", SpanKind.LLM) as span:
            span.set_model(self.model_name)

            oai_tools = None
            if tools:
                oai_tools = []
                for t in tools:
                    oai_tools.append({
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {}),
                        }
                    })

            clean_messages = []
            for m in messages:
                if m.get("role") == "tool" and not m.get("tool_call_id"):
                    clean_messages.append({
                        "role": "user",
                        "content": f"[Tool Result]:\n{m.get('content', '')}"
                    })
                else:
                    clean_messages.append(m)

            payload: dict[str, Any] = {
                "model": self.model_name,
                "messages": clean_messages,
                "temperature": 0.2,
            }
            if oai_tools:
                payload["tools"] = oai_tools

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                **self.extra_headers,
            }

            endpoint = f"{self.base_url}/chat/completions"
            fallback_models = [self.model_name]
            if self.provider_name == "Groq":
                for fb in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "groq/compound", "meta-llama/llama-4-scout-17b-16e-instruct"]:
                    if fb not in fallback_models:
                        fallback_models.append(fb)

            data = None
            try:
                for idx, candidate_m in enumerate(fallback_models):
                    payload["model"] = candidate_m
                    resp = requests.post(endpoint, json=payload, headers=headers, timeout=40)
                    if resp.status_code in (400, 404) and idx < len(fallback_models) - 1:
                        # Model decommissioned or not accessible; try next active candidate
                        continue
                    if resp.status_code != 200:
                        err_text = resp.text[:300]
                        try:
                            err_json = resp.json()
                            err_text = err_json.get("error", {}).get("message", err_text)
                        except Exception:
                            pass
                        raise RuntimeError(f"{self.provider_name} API Error ({resp.status_code}): {err_text}")

                    data = resp.json()
                    self.model_name = candidate_m
                    span.set_model(candidate_m)
                    break
                choice = data["choices"][0]
                message = choice.get("message", {})
                content = message.get("content") or ""
                tool_calls = []

                if message.get("tool_calls"):
                    for tc in message["tool_calls"]:
                        fn = tc.get("function", {})
                        args = fn.get("arguments", "{}")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except Exception:
                                args = {}
                        tool_calls.append(ToolCall(
                            name=fn.get("name", ""),
                            arguments=args,
                        ))

                usage = data.get("usage", {})
                input_tokens = usage.get("prompt_tokens", self._estimate_tokens(json.dumps(messages)))
                output_tokens = usage.get("completion_tokens", self._estimate_tokens(content))

                span.set_tokens(input_tokens, output_tokens)
                span.set_attr("task", task)

                return LLMResponse(
                    content=content,
                    tool_calls=tool_calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self.model_name,
                    raw=data,
                )
            except Exception as e:
                span.set_error(str(e))
                raise


# ---------------------------------------------------------------------------
# OpenAILLM — OpenAI GPT
# ---------------------------------------------------------------------------

class OpenAILLM(OpenAICompatibleLLM):
    """OpenAI GPT models (gpt-4o-mini, gpt-4o, etc.)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url="https://api.openai.com/v1",
            provider_name="OpenAI",
            env_key_name="OPENAI_API_KEY",
        )


# ---------------------------------------------------------------------------
# GroqLLM — Ultra-fast Free Tier (Llama 3.3 70B & 3.1 8B @ 500+ tok/s)
# ---------------------------------------------------------------------------

class GroqLLM(OpenAICompatibleLLM):
    """
    Groq ultra-fast inference with generous free tier.
    Free keys available in 1-click at https://console.groq.com/keys
    """

    def __init__(self, model: str = "openai/gpt-oss-120b", api_key: str | None = None):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            provider_name="Groq",
            env_key_name="GROQ_API_KEY",
        )


# ---------------------------------------------------------------------------
# OpenRouterLLM — Free Tier Open-Source Models
# ---------------------------------------------------------------------------

class OpenRouterLLM(OpenAICompatibleLLM):
    """
    OpenRouter gateway supporting free community models.
    Keys available at https://openrouter.ai/keys
    """

    def __init__(self, model: str = "meta-llama/llama-3.3-70b-instruct:free", api_key: str | None = None):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            provider_name="OpenRouter",
            env_key_name="OPENROUTER_API_KEY",
            extra_headers={
                "HTTP-Referer": "https://prisms.streamlit.app",
                "X-Title": "Prism Glass Box Agent",
            },
        )


# ---------------------------------------------------------------------------
# AnthropicLLM — Claude
# ---------------------------------------------------------------------------

class AnthropicLLM(LLMInterface):
    """Anthropic Claude via the anthropic SDK."""

    model_name = "claude-haiku-3.5"

    def __init__(self, model: str = "claude-haiku-3.5", api_key: str | None = None):
        self.model_name = model
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set")

        import anthropic
        self._client = anthropic.Anthropic(api_key=self._api_key)

    def run(
        self,
        task: str,
        messages: list[dict[str, str]],
        tools: list[dict] | None,
        tracer: Tracer,
    ) -> LLMResponse:
        with tracer.span(f"llm:{task}", SpanKind.LLM) as span:
            span.set_model(self.model_name)

            # Separate system message
            system_text = ""
            api_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_text = msg.get("content", "")
                else:
                    api_messages.append(msg)

            # Build Anthropic tools format
            ant_tools = None
            if tools:
                ant_tools = []
                for t in tools:
                    ant_tools.append({
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "input_schema": t.get("parameters", {}),
                    })

            try:
                kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": api_messages,
                    "max_tokens": 2048,
                }
                if system_text:
                    kwargs["system"] = system_text
                if ant_tools:
                    kwargs["tools"] = ant_tools

                response = self._client.messages.create(**kwargs)

                content = ""
                tool_calls = []

                for block in response.content:
                    if block.type == "text":
                        content += block.text
                    elif block.type == "tool_use":
                        tool_calls.append(ToolCall(
                            name=block.name,
                            arguments=block.input,
                        ))

                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens

                span.set_tokens(input_tokens, output_tokens)
                span.set_attr("task", task)

                return LLMResponse(
                    content=content,
                    tool_calls=tool_calls,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=self.model_name,
                    raw=response,
                )

            except Exception as e:
                span.set_error(str(e))
                raise


# ---------------------------------------------------------------------------
# Factory — auto-detect from env or CLI flag
# ---------------------------------------------------------------------------

def create_llm(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> LLMInterface:
    """
    Create an LLM implementation.

    Priority:
      1. Explicit provider argument
      2. Environment variable PRISM_LLM_PROVIDER
      3. Auto-detect from available API keys
      4. Fall back to DeterministicLLM
    """
    provider = provider or os.environ.get("PRISM_LLM_PROVIDER", "").lower()

    if provider == "groq":
        return GroqLLM(model=model or "openai/gpt-oss-120b", api_key=api_key)
    elif provider == "openrouter":
        return OpenRouterLLM(model=model or "meta-llama/llama-3.3-70b-instruct:free", api_key=api_key)
    elif provider == "gemini":
        return GeminiLLM(model=model or "gemini-3.6-flash", api_key=api_key)
    elif provider == "openai":
        return OpenAILLM(model=model or "gpt-4o-mini", api_key=api_key)
    elif provider == "anthropic":
        return AnthropicLLM(model=model or "claude-haiku-3.5", api_key=api_key)
    elif provider in ("deterministic", "rule", "stub"):
        return DeterministicLLM()

    # If api_key provided without explicit provider, detect by format:
    if api_key:
        if api_key.startswith("gsk_"):
            return GroqLLM(model=model or "openai/gpt-oss-120b", api_key=api_key)
        elif api_key.startswith("sk-or-"):
            return OpenRouterLLM(model=model or "meta-llama/llama-3.3-70b-instruct:free", api_key=api_key)
        elif api_key.startswith("AIza") or api_key.startswith("AQ."):
            return GeminiLLM(model=model or "gemini-3.6-flash", api_key=api_key)
        elif api_key.startswith("sk-ant-"):
            return AnthropicLLM(model=model or "claude-haiku-3.5", api_key=api_key)
        else:
            return OpenAILLM(model=model or "gpt-4o-mini", api_key=api_key)

    # Auto-detect from available environment variables
    if os.environ.get("GROQ_API_KEY"):
        return GroqLLM(model=model or "openai/gpt-oss-120b")
    if os.environ.get("OPENROUTER_API_KEY"):
        return OpenRouterLLM(model=model or "meta-llama/llama-3.3-70b-instruct:free")
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return GeminiLLM(model=model or "gemini-3.6-flash")
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAILLM(model=model or "gpt-4o-mini")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicLLM(model=model or "claude-haiku-3.5")

    # Default: deterministic stub
    return DeterministicLLM()
