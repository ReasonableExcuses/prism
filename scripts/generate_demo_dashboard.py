"""
scripts/generate_demo_dashboard.py — Generate static dashboard with rich demo data for Netlify.
"""

from __future__ import annotations

import sys
import shutil
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from prism.trace import Tracer
from prism.llm import DeterministicLLM
from prism.tools import ToolRegistry
from prism.context import ContextEngine
from prism.agent import Agent
from prism.dashboard import generate_dashboard


def main():
    print("🔮 Generating rich demo trace for static dashboard...")
    tracer = Tracer(artifacts_dir="artifacts/demo", run_id="demo_glassbox_v1")
    tools = ToolRegistry(tracer)
    context = ContextEngine(tracer=tracer)
    llm = DeterministicLLM()
    agent = Agent(llm, tools, context, tracer)

    # Turn 1: Arithmetic & Knowledge
    print("  Turn 1: Math calculation...")
    agent.run("Calculate 25 * 48 and tell me about python programming.")

    # Turn 2: Search & Summarize
    print("  Turn 2: Search query...")
    agent.run("Search for latest developments in artificial intelligence agent observability.")

    # Turn 3: Error injection & recovery
    print("  Turn 3: Trigger tool failure and recovery...")
    tools._fail_search = True
    agent.run("Search for quantum computing breakthroughs.")
    tools._fail_search = False

    # Turn 4: Context recall & synthesis
    print("  Turn 4: Multi-turn synthesis...")
    agent.run("Summarize everything we have explored in this session.")

    # Generate dashboard
    public_dir = _root / "public"
    public_dir.mkdir(exist_ok=True)

    dash_path = generate_dashboard(tracer, context, agent, output_dir=str(public_dir))
    index_path = public_dir / "index.html"
    shutil.copy(dash_path, index_path)

    print(f"✅ Generated static dashboard at {index_path} ({index_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
