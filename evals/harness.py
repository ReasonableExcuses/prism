"""
evals/harness.py — Test infrastructure for the Prism eval suite.

Provides a controlled environment for running scenarios against the
agent with the DeterministicLLM.  Each scenario is a function that
exercises one aspect of the system and makes assertions.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any

# Ensure project root is on path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from prism.trace import Tracer
from prism.llm import DeterministicLLM
from prism.tools import ToolRegistry
from prism.context import ContextEngine
from prism.agent import Agent


# ---------------------------------------------------------------------------
# Scenario result
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    name: str
    passed: bool
    assertions_total: int
    assertions_passed: int
    assertions_failed: int
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class Harness:
    """
    Controlled test environment for Prism scenarios.

    Creates a fresh agent with DeterministicLLM for each scenario.
    Provides assertion helpers that track pass/fail counts.
    """

    def __init__(self):
        self.tracer: Tracer = None
        self.agent: Agent = None
        self.tools: ToolRegistry = None
        self.context: ContextEngine = None
        self.llm: DeterministicLLM = None
        self._assertions_total = 0
        self._assertions_passed = 0
        self._assertions_failed = 0
        self._errors: list[str] = []

    def setup(self, fail_search: bool = False, fail_url: bool = False) -> None:
        """Create a fresh agent for a scenario."""
        self.tracer = Tracer(artifacts_dir="artifacts/test", run_id=f"test_{int(time.time()*1000)}")
        self.llm = DeterministicLLM()
        self.tools = ToolRegistry(self.tracer, fail_search=fail_search, fail_url=fail_url)
        self.context = ContextEngine(tracer=self.tracer)
        self.agent = Agent(self.llm, self.tools, self.context, self.tracer)
        self._assertions_total = 0
        self._assertions_passed = 0
        self._assertions_failed = 0
        self._errors = []

    def query(self, text: str) -> str:
        """Send a query through the agent and return the response."""
        response = self.agent.run(text)
        return response.content

    def assert_true(self, condition: bool, message: str) -> None:
        """Assert a condition is true."""
        self._assertions_total += 1
        if condition:
            self._assertions_passed += 1
        else:
            self._assertions_failed += 1
            self._errors.append(f"FAIL: {message}")

    def assert_contains(self, text: str, substring: str, message: str = "") -> None:
        """Assert that text contains a substring (case-insensitive)."""
        msg = message or f"Expected '{substring}' in response"
        self.assert_true(substring.lower() in text.lower(), msg)

    def assert_not_contains(self, text: str, substring: str, message: str = "") -> None:
        """Assert that text does NOT contain a substring."""
        msg = message or f"Expected '{substring}' NOT in response"
        self.assert_true(substring.lower() not in text.lower(), msg)

    def assert_gt(self, value: float, threshold: float, message: str = "") -> None:
        """Assert value > threshold."""
        msg = message or f"Expected {value} > {threshold}"
        self.assert_true(value > threshold, msg)

    def assert_eq(self, actual: Any, expected: Any, message: str = "") -> None:
        """Assert equality."""
        msg = message or f"Expected {actual} == {expected}"
        self.assert_true(actual == expected, msg)


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def run_scenario(name: str, fn: Callable[[Harness], None], verbose: bool = False) -> ScenarioResult:
    """Run a single scenario and return the result."""
    harness = Harness()
    start = time.time()

    try:
        fn(harness)
    except Exception as e:
        harness._assertions_failed += 1
        harness._errors.append(f"EXCEPTION: {type(e).__name__}: {e}")

    duration_ms = (time.time() - start) * 1000
    passed = harness._assertions_failed == 0

    result = ScenarioResult(
        name=name,
        passed=passed,
        assertions_total=harness._assertions_total,
        assertions_passed=harness._assertions_passed,
        assertions_failed=harness._assertions_failed,
        errors=list(harness._errors),
        duration_ms=duration_ms,
    )

    if verbose:
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name} — {harness._assertions_passed}/{harness._assertions_total} "
              f"assertions ({duration_ms:.0f}ms)")
        for err in harness._errors:
            print(f"     ↳ {err}")

    return result
