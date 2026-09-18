"""
prism/cli.py — Interactive REPL for live demo.

Commands:
  :trace     — print the span tree for the last run
  :cost      — cumulative token/cost breakdown
  :context   — current context window allocation
  :dashboard — generate and open the HTML dashboard
  :fail      — trigger a deliberate tool failure to demo recovery
  :unfail    — restore tool reliability
  :history   — conversation history with summaries
  :notes     — show saved notes
  :help      — list commands
  :quit      — exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from pathlib import Path


def _setup_path():
    """Ensure the project root is on sys.path."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

_setup_path()

from prism.trace import Tracer
from prism.llm import create_llm
from prism.tools import ToolRegistry
from prism.context import ContextEngine
from prism.agent import Agent


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_BOLD    = "\033[1m"
_DIM     = "\033[2m"
_CYAN    = "\033[36m"
_GREEN   = "\033[32m"
_YELLOW  = "\033[33m"
_RED     = "\033[31m"
_MAGENTA = "\033[35m"
_RESET   = "\033[0m"


def _header(text: str) -> str:
    return f"\n{_BOLD}{_CYAN}{'═' * 60}{_RESET}\n  {_BOLD}{text}{_RESET}\n{_BOLD}{_CYAN}{'═' * 60}{_RESET}"


def _print_cost_table(tracer: Tracer) -> None:
    """Print a formatted cost/token breakdown."""
    s = tracer.summary()
    print(_header("💰 Cost & Token Summary"))
    print(f"  {'LLM Calls:':<20} {s['llm_calls']}")
    print(f"  {'Tool Calls:':<20} {s['tool_calls']}")
    print(f"  {'Input Tokens:':<20} {s['total_input_tokens']:,}")
    print(f"  {'Output Tokens:':<20} {s['total_output_tokens']:,}")
    print(f"  {'Total Tokens:':<20} {s['total_tokens']:,}")
    print(f"  {'Cost (USD):':<20} ${s['cost_usd']:.6f}")
    print(f"  {'Wall Time:':<20} {s['wall_ms']:.0f}ms")
    print(f"  {'Errors:':<20} {s['errors']}")
    print(f"  {'Spans:':<20} {s['span_count']}")
    print()


def _print_context_status(context: ContextEngine) -> None:
    """Print the context engine status."""
    status = context.get_status()
    print(_header("📦 Context Window Status"))
    print(f"  {'Total Turns:':<25} {status['total_turns']}")
    print(f"  {'Stored Turns:':<25} {status['stored_turns']}")
    print(f"  {'Window Size:':<25} {status['window_size']}")
    print(f"  {'Key Facts:':<25} {status['key_facts']}")
    print(f"  {'Has Summary:':<25} {'Yes' if status['has_summary'] else 'No'}")
    if status['has_summary']:
        print(f"  {'Summary Length:':<25} {status['summary_length']} chars")

    # Budget
    b = status['budget']
    print(f"\n  {_BOLD}Token Budget:{_RESET}")
    print(f"  {'  System Prompt:':<25} {b['system_prompt']}")
    print(f"  {'  Key Facts:':<25} {b['key_facts']}")
    print(f"  {'  Rolling Summary:':<25} {b['rolling_summary']}")
    print(f"  {'  Recent Turns:':<25} {b['recent_turns']}")
    print(f"  {'  Tool Results:':<25} {b['tool_results']}")
    print(f"  {'  Response Room:':<25} {b['response_room']}")
    print(f"  {'  Total:':<25} {b['total']}")

    # Allocation history
    history = context.get_allocation_history()
    if history:
        latest = history[-1]
        print(f"\n  {_BOLD}Latest Allocation:{_RESET}")
        for section, tokens in latest['allocation'].items():
            bar_len = min(30, tokens // 20)
            bar = "█" * bar_len
            print(f"  {'  ' + section + ':':<25} {tokens:>5} tok  {_CYAN}{bar}{_RESET}")
        print(f"  {'  Utilization:':<25} {latest['utilization_pct']:.1f}%")
    print()


def _print_history(context: ContextEngine) -> None:
    """Print conversation history."""
    print(_header("📜 Conversation History"))
    status = context.get_status()
    print(f"  Total turns: {status['total_turns']}")
    if status['has_summary']:
        print(f"\n  {_DIM}Rolling Summary:{_RESET}")
        # Access internal state for display
        if context._rolling_summary:
            for line in context._rolling_summary.split("\n"):
                print(f"    {_DIM}{line}{_RESET}")
    print(f"\n  {_BOLD}Recent Turns (in window):{_RESET}")
    recent = context._turns[-context.window_size:]
    for turn in recent:
        icon = "👤" if turn.role == "user" else "🤖" if turn.role == "assistant" else "🔧"
        color = _GREEN if turn.role == "user" else _CYAN if turn.role == "assistant" else _YELLOW
        preview = turn.content[:120].replace("\n", " ")
        print(f"    {icon} {color}[{turn.turn_number}] {preview}{_RESET}")
    print()


def _print_help() -> None:
    """Print the help message."""
    print(_header("🔮 Prism Commands"))
    commands = [
        (":trace",     "Show the span tree for the current session"),
        (":cost",      "Show cumulative token/cost breakdown"),
        (":context",   "Show context window allocation and status"),
        (":dashboard", "Generate and open the interactive HTML dashboard"),
        (":fail",      "Trigger deliberate tool failures (for demo)"),
        (":unfail",    "Restore tool reliability"),
        (":history",   "Show conversation history with summaries"),
        (":notes",     "Show saved research notes"),
        (":help",      "Show this help message"),
        (":quit",      "Exit Prism"),
    ]
    for cmd, desc in commands:
        print(f"  {_YELLOW}{cmd:<15}{_RESET} {desc}")
    print()


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="🔮 Prism — The Glass Box AI Research Agent")
    parser.add_argument("--llm", type=str, default=None,
                        help="LLM provider: groq, openrouter, gemini, openai, anthropic, deterministic")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model name (e.g., llama-3.3-70b-versatile, gemini-3.6-flash, gpt-4o-mini)")
    parser.add_argument("--fail-search", action="store_true",
                        help="Start with search tool deliberately failing")
    parser.add_argument("--fail-url", action="store_true",
                        help="Start with URL reader deliberately failing")
    parser.add_argument("--max-steps", type=int, default=5,
                        help="Maximum reasoning steps per query")
    args = parser.parse_args()

    # Initialise components
    tracer = Tracer(artifacts_dir="artifacts")
    llm = create_llm(provider=args.llm, model=args.model)
    tools = ToolRegistry(tracer, fail_search=args.fail_search, fail_url=args.fail_url)
    context = ContextEngine(tracer=tracer)
    agent = Agent(llm, tools, context, tracer, max_steps=args.max_steps)

    # Welcome
    print(f"\n{_BOLD}{_MAGENTA}{'═' * 60}{_RESET}")
    print(f"  {_BOLD}🔮 PRISM — The Glass Box AI Research Agent{_RESET}")
    print(f"  {_DIM}Every decision traced. Every token counted.{_RESET}")
    print(f"  {_DIM}Model: {llm.model_name}  |  Max steps: {args.max_steps}{_RESET}")
    print(f"{_BOLD}{_MAGENTA}{'═' * 60}{_RESET}")
    print(f"  Type {_YELLOW}:help{_RESET} for commands, or just start chatting.\n")

    while True:
        try:
            user_input = input(f"{_GREEN}you ❯ {_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_DIM}Goodbye!{_RESET}")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith(":"):
            cmd = user_input.lower().split()[0]

            if cmd in (":quit", ":exit", ":q"):
                trace_path = tracer.save()
                print(f"\n{_DIM}Trace saved to {trace_path}{_RESET}")
                print(f"{_DIM}Goodbye!{_RESET}")
                break

            elif cmd == ":trace":
                tracer.print_tree()

            elif cmd == ":cost":
                _print_cost_table(tracer)

            elif cmd == ":context":
                _print_context_status(context)

            elif cmd == ":dashboard":
                try:
                    from prism.dashboard import generate_dashboard
                    path = generate_dashboard(tracer, context, agent)
                    print(f"  {_GREEN}Dashboard generated: {path}{_RESET}")
                    webbrowser.open(f"file://{path.resolve()}")
                except Exception as e:
                    print(f"  {_RED}Dashboard error: {e}{_RESET}")

            elif cmd == ":fail":
                tools._fail_search = True
                tools._fail_url = True
                print(f"  {_YELLOW}⚠ Tool failures enabled. Search and URL tools will fail.{_RESET}")
                print(f"  {_DIM}Try a query to see error recovery in action.{_RESET}")

            elif cmd == ":unfail":
                tools._fail_search = False
                tools._fail_url = False
                print(f"  {_GREEN}✓ Tool failures disabled. All tools operational.{_RESET}")

            elif cmd == ":history":
                _print_history(context)

            elif cmd == ":notes":
                notes = tools.get_notes_summary()
                print(f"\n  {notes}\n")

            elif cmd == ":help":
                _print_help()

            else:
                print(f"  {_RED}Unknown command: {cmd}. Type :help for available commands.{_RESET}")

            continue

        # Regular query — run through the agent
        print(f"\n  {_DIM}🔮 Thinking...{_RESET}")
        try:
            response = agent.run(user_input)

            # Print the response
            print(f"\n{_CYAN}prism ❯{_RESET} {response.content}")

            # Print step summary
            tool_steps = [s for s in response.steps if s.action == "tool_call"]
            error_steps = [s for s in response.steps if s.action == "error_recovery"]

            footer_parts = []
            footer_parts.append(f"{response.total_steps} steps")
            if tool_steps:
                tool_names = [s.tool_name for s in tool_steps]
                footer_parts.append(f"tools: {', '.join(tool_names)}")
            if error_steps:
                footer_parts.append(f"{_RED}{len(error_steps)} error(s) recovered{_RESET}")
            footer_parts.append(f"{response.tokens_used:,} tokens")
            footer_parts.append(f"${response.cost_usd:.6f}")
            footer_parts.append(f"{response.duration_ms:.0f}ms")

            print(f"  {_DIM}{' | '.join(footer_parts)}{_RESET}\n")

        except Exception as e:
            print(f"\n  {_RED}Error: {e}{_RESET}")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()
