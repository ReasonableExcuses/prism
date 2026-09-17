"""
evals/run_evals.py — Run the Prism eval suite.

Usage:
    python evals/run_evals.py             # all scenarios
    python evals/run_evals.py -v          # verbose
    python evals/run_evals.py --only s04  # single scenario
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from evals.harness import run_scenario
from evals.scenarios import SCENARIOS


def main():
    parser = argparse.ArgumentParser(description="🔮 Prism Eval Suite")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-assertion detail")
    parser.add_argument("--only", type=str, default=None, help="Run only scenarios matching this substring")
    args = parser.parse_args()

    print(f"\n{'═' * 60}")
    print(f"  🔮 PRISM EVAL SUITE")
    print(f"{'═' * 60}\n")

    # Filter scenarios
    to_run = SCENARIOS
    if args.only:
        to_run = [(n, f) for n, f in SCENARIOS if args.only in n]
        if not to_run:
            print(f"  No scenarios matching '{args.only}'")
            sys.exit(1)

    results = []
    start_time = time.time()

    for name, fn in to_run:
        result = run_scenario(name, fn, verbose=args.verbose)
        results.append(result)

    total_ms = (time.time() - start_time) * 1000
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total_assertions = sum(r.assertions_total for r in results)
    passed_assertions = sum(r.assertions_passed for r in results)

    print(f"\n{'─' * 60}")

    # Summary
    if failed == 0:
        status = "✅ ALL PASSED"
    else:
        status = f"❌ {failed} FAILED"

    print(f"\n  {status}")
    print(f"  Scenarios:  {passed}/{len(results)}")
    print(f"  Assertions: {passed_assertions}/{total_assertions}")
    print(f"  Time:       {total_ms:.0f}ms")
    print(f"\n{'═' * 60}\n")

    # Save results
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    results_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenarios_total": len(results),
        "scenarios_passed": passed,
        "scenarios_failed": failed,
        "assertions_total": total_assertions,
        "assertions_passed": passed_assertions,
        "duration_ms": round(total_ms, 1),
        "results": [
            {
                "name": r.name,
                "passed": r.passed,
                "assertions": f"{r.assertions_passed}/{r.assertions_total}",
                "duration_ms": round(r.duration_ms, 1),
                "errors": r.errors,
            }
            for r in results
        ],
    }
    (out_dir / "eval_results.json").write_text(
        json.dumps(results_data, indent=2), encoding="utf-8"
    )

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
