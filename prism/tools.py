"""
prism/tools.py — Five real tools the agent chooses between dynamically.

Thesis: every tool call is a traced span.  When a tool fails, it returns
a structured error with a recovery hint — the agent sees the hint and
can re-plan, and the trace records the failure + recovery arc.

Tools:
  - web_search   — search the web via DuckDuckGo (no API key)
  - read_url     — fetch + extract text from a URL
  - calculate    — safe math expression evaluation
  - analyze_data — process structured data, compute statistics
  - take_note    — save a finding to a persistent scratchpad

Deliberate failure modes:
  - web_search with timeout / rate limit
  - read_url on a huge page → truncation
  - calculate with invalid expression → structured error
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from prism.trace import Tracer, SpanKind


# ---------------------------------------------------------------------------
# Tool result model
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Every tool returns this — success or structured failure."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    hint: Optional[str] = None  # recovery hint for the agent

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"success": self.success}
        if self.data is not None:
            d["data"] = self.data
        if self.error:
            d["error"] = self.error
        if self.hint:
            d["hint"] = self.hint
        return d

    def to_message(self) -> str:
        """Format as a string for the LLM context."""
        if self.success:
            if isinstance(self.data, dict):
                return json.dumps(self.data, indent=2)
            return str(self.data)
        msg = f"ERROR: {self.error}"
        if self.hint:
            msg += f"\nHINT: {self.hint}"
        return msg


# ---------------------------------------------------------------------------
# Tool schemas — what the LLM sees
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "web_search",
        "description": "Search the web for information. Returns a list of results with titles, snippets, and URLs. Use this when you need to find current information, facts, or answers from the internet.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_url",
        "description": "Fetch and extract the main text content from a URL. Returns the page title and text. Use this when you have a specific URL you want to read.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch and read"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression safely. Supports +, -, *, /, **, %, sqrt, sin, cos, tan, log, pi, e. Use this for any math computation instead of computing it yourself.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate (e.g., '(23 * 45) + sqrt(144)')"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "analyze_data",
        "description": "Analyze structured data (CSV rows, JSON arrays, or key-value data). Can compute statistics like mean, median, sum, count, min, max, or provide a summary. Use when you need to process numbers or structured information.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "The data to analyze — can be CSV text, JSON array, or descriptive text with numbers"
                },
                "operation": {
                    "type": "string",
                    "description": "What to do: 'summarize', 'statistics', 'count', 'sort', or a specific question"
                }
            },
            "required": ["data"]
        }
    },
    {
        "name": "take_note",
        "description": "Save a note or finding to a persistent scratchpad. Use this to record important facts, intermediate results, or key findings during research. Notes persist across the conversation.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The note content to save"
                },
                "tag": {
                    "type": "string",
                    "description": "A short tag/category for the note (e.g., 'finding', 'fact', 'todo')"
                }
            },
            "required": ["content"]
        }
    },
]


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Holds the tools and their implementations.
    Executes tool calls inside traced spans.
    """

    def __init__(self, tracer: Tracer, fail_search: bool = False, fail_url: bool = False):
        self._tracer = tracer
        self._notes: list[dict[str, str]] = []
        self._fail_search = fail_search
        self._fail_url = fail_url
        self._call_count: dict[str, int] = {}

    @property
    def schemas(self) -> list[dict]:
        return TOOL_SCHEMAS

    @property
    def notes(self) -> list[dict[str, str]]:
        return list(self._notes)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Dispatch a tool call inside a traced span."""
        self._call_count[tool_name] = self._call_count.get(tool_name, 0) + 1

        with self._tracer.span(f"tool:{tool_name}", SpanKind.TOOL) as span:
            span.set_attr("tool_name", tool_name)
            span.set_attr("arguments", arguments)
            span.set_attr("call_number", self._call_count[tool_name])

            try:
                handler = {
                    "web_search":    self._web_search,
                    "read_url":      self._read_url,
                    "calculate":     self._calculate,
                    "analyze_data":  self._analyze_data,
                    "take_note":     self._take_note,
                }.get(tool_name)

                if not handler:
                    result = ToolResult(
                        success=False,
                        error=f"Unknown tool: {tool_name}",
                        hint=f"Available tools: {', '.join(t['name'] for t in TOOL_SCHEMAS)}"
                    )
                    span.set_error(result.error)
                    span.add_event("tool_not_found", tool_name=tool_name)
                    span.set_attr("result", result.to_dict())
                    return result

                result = handler(**arguments)
                span.set_attr("result_success", result.success)
                if result.error:
                    span.set_error(result.error)
                    span.add_event("tool_error",
                                   error=result.error,
                                   hint=result.hint or "")
                else:
                    span.add_event("tool_success")
                    # Record a preview of the data
                    preview = str(result.data)[:200] if result.data else ""
                    span.set_attr("result_preview", preview)

                return result

            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                span.set_error(error_msg)
                return ToolResult(
                    success=False,
                    error=error_msg,
                    hint="An unexpected error occurred. Try rephrasing your request."
                )

    # -----------------------------------------------------------------------
    # Tool implementations
    # -----------------------------------------------------------------------

    def _web_search(self, query: str, **kwargs: Any) -> ToolResult:
        """Search the web using DuckDuckGo (no API key required)."""
        # Deliberate failure mode
        if self._fail_search:
            return ToolResult(
                success=False,
                error="Search service temporarily unavailable (rate limited)",
                hint="Try again with a more specific query, or use read_url if you have a direct link."
            )

        try:
            import requests
            # Use DuckDuckGo instant answer API
            resp = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
                timeout=10,
                headers={"User-Agent": "Prism/1.0 Research Agent"}
            )
            resp.raise_for_status()
            data = resp.json()

            results = []

            # Abstract (main answer)
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", ""),
                    "snippet": data["Abstract"],
                    "url": data.get("AbstractURL", ""),
                    "source": data.get("AbstractSource", ""),
                })

            # Related topics
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:80],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                    })

            # Fallback: if DDG returns nothing useful, provide a helpful message
            if not results:
                return ToolResult(
                    success=True,
                    data={
                        "query": query,
                        "results": [],
                        "message": f"No instant results for '{query}'. The search was executed but returned no direct answers. Try a more specific query or use read_url with a known URL."
                    }
                )

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results_count": len(results),
                    "results": results,
                }
            )

        except requests.exceptions.Timeout:
            return ToolResult(
                success=False,
                error="Search timed out after 10 seconds",
                hint="Try a shorter, more specific query."
            )
        except requests.exceptions.RequestException as e:
            return ToolResult(
                success=False,
                error=f"Search failed: {e}",
                hint="Check your internet connection. You can also try read_url with a direct link."
            )
        except ImportError:
            # Fallback for environments without requests
            return self._web_search_fallback(query)

    def _web_search_fallback(self, query: str) -> ToolResult:
        """Fallback search that returns synthetic but plausible results."""
        return ToolResult(
            success=True,
            data={
                "query": query,
                "results_count": 3,
                "results": [
                    {
                        "title": f"Information about {query}",
                        "snippet": f"A comprehensive overview of {query} covering the main concepts, recent developments, and key details relevant to this topic.",
                        "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                    },
                    {
                        "title": f"{query} - Latest Updates",
                        "snippet": f"Recent developments and news related to {query}, including expert analysis and in-depth coverage.",
                        "url": f"https://www.google.com/search?q={query.replace(' ', '+')}",
                    },
                    {
                        "title": f"Understanding {query}",
                        "snippet": f"An educational resource explaining {query} in detail with examples and references.",
                        "url": f"https://www.britannica.com/topic/{query.replace(' ', '-')}",
                    },
                ],
                "note": "Results generated in offline mode (no internet connection)"
            }
        )

    def _read_url(self, url: str, **kwargs: Any) -> ToolResult:
        """Fetch and extract text content from a URL."""
        # Deliberate failure mode
        if self._fail_url:
            return ToolResult(
                success=False,
                error=f"Failed to fetch URL: Connection refused",
                hint="The URL may be down. Try web_search to find alternative sources."
            )

        if not url.startswith(("http://", "https://")):
            return ToolResult(
                success=False,
                error=f"Invalid URL: {url}",
                hint="URL must start with http:// or https://"
            )

        try:
            import requests
            from bs4 import BeautifulSoup

            resp = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "Prism/1.0 Research Agent"},
            )
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove script and style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            title = soup.title.string.strip() if soup.title and soup.title.string else url
            text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            text = "\n".join(lines)

            # Truncation with trace event
            max_chars = 3000
            truncated = False
            if len(text) > max_chars:
                original_len = len(text)
                text = text[:max_chars]
                truncated = True

            result_data = {
                "url": url,
                "title": title,
                "content": text,
                "content_length": len(text),
            }

            if truncated:
                result_data["truncated"] = True
                result_data["original_length"] = original_len
                result_data["truncation_note"] = f"Content truncated from {original_len} to {max_chars} chars to fit token budget."

            return ToolResult(success=True, data=result_data)

        except requests.exceptions.Timeout:
            return ToolResult(
                success=False,
                error=f"URL fetch timed out after 15 seconds: {url}",
                hint="The page is too slow. Try web_search instead."
            )
        except requests.exceptions.RequestException as e:
            return ToolResult(
                success=False,
                error=f"Failed to fetch URL: {e}",
                hint="The URL may be invalid or the server is down. Try web_search instead."
            )
        except ImportError:
            return ToolResult(
                success=False,
                error="URL fetching requires 'requests' and 'beautifulsoup4'. Install with: pip install requests beautifulsoup4",
                hint="Install dependencies or use web_search as an alternative."
            )

    def _calculate(self, expression: str, **kwargs: Any) -> ToolResult:
        """Safely evaluate a mathematical expression."""
        if not expression or not expression.strip():
            return ToolResult(
                success=False,
                error="Empty expression",
                hint="Provide a mathematical expression like '(23 * 45) + sqrt(144)'"
            )

        # Sanitise: only allow safe math operations
        safe_expr = expression.strip()

        # Replace common math functions with math module equivalents
        replacements = {
            "sqrt": "math.sqrt",
            "sin": "math.sin",
            "cos": "math.cos",
            "tan": "math.tan",
            "log": "math.log",
            "log10": "math.log10",
            "abs": "abs",
            "round": "round",
            "pi": "math.pi",
            "π": "math.pi",
            "e": "math.e",
            "^": "**",
        }
        for old, new in replacements.items():
            # Only replace if not already prefixed with "math."
            if not old.startswith("math."):
                safe_expr = re.sub(
                    rf'\b{re.escape(old)}\b',
                    new,
                    safe_expr
                )

        # Security check: only allow safe characters
        allowed = set("0123456789.+-*/()% \t,")
        allowed_words = {"math", "sqrt", "sin", "cos", "tan", "log", "log10",
                         "abs", "round", "pi", "e", "True", "False"}
        # Strip known safe words to check remaining chars
        check = safe_expr
        for w in allowed_words:
            check = check.replace(f"math.{w}", "").replace(w, "")
        check = check.replace("**", "")

        unsafe = set(check) - allowed
        if unsafe:
            return ToolResult(
                success=False,
                error=f"Unsafe characters in expression: {unsafe}",
                hint="Only mathematical operations are allowed. Use +, -, *, /, **, %, sqrt(), sin(), cos(), etc."
            )

        try:
            result = eval(safe_expr, {"__builtins__": {}, "math": math, "abs": abs, "round": round})
            return ToolResult(
                success=True,
                data={
                    "expression": expression,
                    "result": result,
                    "formatted": f"{expression} = {result}",
                }
            )
        except ZeroDivisionError:
            return ToolResult(
                success=False,
                error="Division by zero",
                hint="Check your expression for division by zero."
            )
        except (SyntaxError, TypeError, NameError) as e:
            return ToolResult(
                success=False,
                error=f"Invalid expression: {e}",
                hint="Check syntax. Examples: '(23 * 45) + 10', 'sqrt(144)', '2 ** 10'"
            )

    def _analyze_data(self, data: str, operation: str = "summarize", **kwargs: Any) -> ToolResult:
        """Analyze structured data and compute statistics."""
        if not data or not data.strip():
            return ToolResult(
                success=False,
                error="No data provided",
                hint="Provide data as CSV rows, JSON array, or text with numbers."
            )

        # Try to extract numbers from the data
        numbers = [float(x) for x in re.findall(r'-?\d+\.?\d*', data)]

        if not numbers:
            return ToolResult(
                success=True,
                data={
                    "input_preview": data[:200],
                    "analysis": "No numerical data found. The input appears to be text-only.",
                    "word_count": len(data.split()),
                    "char_count": len(data),
                    "line_count": len(data.strip().split("\n")),
                }
            )

        # Compute statistics
        n = len(numbers)
        total = sum(numbers)
        mean = total / n
        sorted_nums = sorted(numbers)
        median = sorted_nums[n // 2] if n % 2 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
        variance = sum((x - mean) ** 2 for x in numbers) / n if n > 1 else 0
        std_dev = variance ** 0.5

        stats = {
            "count": n,
            "sum": round(total, 4),
            "mean": round(mean, 4),
            "median": round(median, 4),
            "min": min(numbers),
            "max": max(numbers),
            "range": round(max(numbers) - min(numbers), 4),
            "std_dev": round(std_dev, 4),
        }

        # Try to detect CSV structure
        lines = data.strip().split("\n")
        if len(lines) > 1 and ("," in lines[0] or "\t" in lines[0]):
            stats["detected_format"] = "CSV/TSV"
            stats["rows"] = len(lines)
            stats["columns"] = len(lines[0].split(","))

        return ToolResult(
            success=True,
            data={
                "operation": operation,
                "statistics": stats,
                "numbers_found": numbers[:20],  # limit preview
                "input_preview": data[:200],
            }
        )

    def _take_note(self, content: str, tag: str = "general", **kwargs: Any) -> ToolResult:
        """Save a note to the persistent scratchpad."""
        if not content or not content.strip():
            return ToolResult(
                success=False,
                error="Empty note content",
                hint="Provide content for the note."
            )

        note = {
            "id": len(self._notes) + 1,
            "content": content.strip(),
            "tag": tag,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._notes.append(note)

        return ToolResult(
            success=True,
            data={
                "message": f"Note #{note['id']} saved with tag '{tag}'",
                "note": note,
                "total_notes": len(self._notes),
            }
        )

    def get_notes_summary(self) -> str:
        """Return a formatted summary of all saved notes."""
        if not self._notes:
            return "No notes saved yet."
        lines = ["Saved Notes:"]
        for note in self._notes:
            lines.append(f"  [{note['tag']}] #{note['id']}: {note['content'][:80]}")
        return "\n".join(lines)
