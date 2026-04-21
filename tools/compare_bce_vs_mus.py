#!/usr/bin/env python3
"""
Compare BCE vs MUS guidance on the actual GNN planning task.

Run from repo root, for example:

python tools/compare_bce_vs_mus.py \
    --domain_name MazeNamo \
    --train_planner_name fd-lama-first \
    --test_planner_name fd-lama-first \
    --planner_type flax \
    --num_seeds 3 \
    --num_train_problems 50 \
    --num_test_problems 50 \
    --train_timeout 120 \
    --test_timeout 120 \
    --num_epochs 301 \
    --out results/bce_vs_mus_mazenamo.json

This script:
1. Runs the repo's existing experiment entrypoint twice:
   - guider_name = gnn-bce-10
   - guider_name = gnn-grape-must
2. Parses the printed task-level metrics:
   - total avg planning time
   - total avg success rate
   - total avg plan length
   - per-seed lists, when available
3. Writes a JSON report and prints a concise comparison.

Decision rule:
- Higher success rate wins first
- If essentially tied, lower avg planning time wins
- If still tied, lower avg plan length wins
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class RunResult:
    guider_name: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    avg_planning_time: float | None
    avg_success_rate: float | None
    avg_plan_length: float | None
    planning_time_list: list[float]
    success_rate_list: list[float]
    plan_length_list: list[float]
    failure_problem_list: list[str]
    failure_problem_set: list[str]


def _parse_float(label: str, text: str) -> float | None:
    patterns = [
        rf"{re.escape(label)}:\s*([+-]?\d+(?:\.\d+)?)",
        rf"{re.escape(label)}\s*=\s*([+-]?\d+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return None


def _parse_python_list(label: str, text: str) -> list[Any]:
    patterns = [
        rf"{re.escape(label)}:\s*(\[[^\n]*\])",
        rf"{re.escape(label)}\s*=\s*(\[[^\n]*\])",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                value = ast.literal_eval(m.group(1))
                if isinstance(value, list):
                    return value
            except Exception:
                pass
    return []


def _parse_result(
    guider_name: str,
    command: list[str],
    completed: subprocess.CompletedProcess[str],
) -> RunResult:
    text_out = completed.stdout
    text_err = completed.stderr

    return RunResult(
        guider_name=guider_name,
        command=command,
        returncode=completed.returncode,
        stdout=text_out,
        stderr=text_err,
        avg_planning_time=_parse_float("total avg planning time", text_out),
        avg_success_rate=_parse_float("total avg success rate", text_out),
        avg_plan_length=_parse_float("total avg plan length", text_out),
        planning_time_list=[float(x) for x in _parse_python_list("planning time list", text_out)],
        success_rate_list=[float(x) for x in _parse_python_list("success rate list", text_out)],
        plan_length_list=[float(x) for x in _parse_python_list("plan length list", text_out)],
        failure_problem_list=[str(x) for x in _parse_python_list("failure problem list", text_out)],
        failure_problem_set=[str(x) for x in _parse_python_list("total failure problem set", text_out)],
    )


def _mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def _stdev(xs: list[float]) -> float | None:
    return statistics.stdev(xs) if len(xs) >= 2 else None


def _safe_float(x: float | None) -> float:
    return float("nan") if x is None else x


def _is_close(a: float | None, b: float | None, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def _winner_label(bce: RunResult, mus: RunResult) -> tuple[str, str]:
    """
    Decide winner using task-first priorities:
    1) success rate higher is better
    2) planning time lower is better
    3) plan length lower is better
    """
    bce_sr = bce.avg_success_rate
    mus_sr = mus.avg_success_rate

    if bce_sr is None or mus_sr is None:
        return "inconclusive", "Could not parse success rates from one or both runs."

    if not _is_close(bce_sr, mus_sr, tol=1e-9):
        if mus_sr > bce_sr:
            return "mus", "MUS has the higher average success rate."
        return "bce", "BCE has the higher average success rate."

    bce_t = bce.avg_planning_time
    mus_t = mus.avg_planning_time
    if bce_t is not None and mus_t is not None and not _is_close(bce_t, mus_t, tol=1e-9):
        if mus_t < bce_t:
            return "mus", "Success rates tie, and MUS has lower average planning time."
        return "bce", "Success rates tie, and BCE has lower average planning time."

    bce_len = bce.avg_plan_length
    mus_len = mus.avg_plan_length
    if bce_len is not None and mus_len is not None and not _is_close(bce_len, mus_len, tol=1e-9):
        if mus_len < bce_len:
            return "mus", "Success rate and time tie, and MUS has shorter average plans."
        return "bce", "Success rate and time tie, and BCE has shorter average plans."

    return "tie", "The compared task metrics are tied under the current decision rule."


def _format_metric(mean_val: float | None, xs: list[float]) -> str:
    if mean_val is None:
        return "N/A"
    sd = _stdev(xs)
    if sd is None:
        return f"{mean_val:.6f}"
    return f"{mean_val:.6f} ± {sd:.6f}"


def _run_one(
    repo_root: Path,
    guider_name: str,
    args: argparse.Namespace,
) -> RunResult:
    cmd = [
        sys.executable,
        str(repo_root / "src" / "main.py"),
        "--domain_name", args.domain_name,
        "--train_planner_name", args.train_planner_name,
        "--test_planner_name", args.test_planner_name,
        "--guider_name", guider_name,
        "--num_seeds", str(args.num_seeds),
        "--num_train_problems", str(args.num_train_problems),
        "--num_test_problems", str(args.num_test_problems),
        "--planner_type", args.planner_type,
        "--train_timeout", str(args.train_timeout),
        "--test_timeout", str(args.test_timeout),
        "--num_epochs", str(args.num_epochs),
        "--cmpl_rules", args.cmpl_rules,
        "--relx_rules", args.relx_rules,
    ]

    print(f"\n=== Running {guider_name} ===")
    print(" ".join(cmd), flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )

    print(completed.stdout, flush=True)
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr, flush=True)

    return _parse_result(guider_name, cmd, completed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name", required=True, type=str)
    parser.add_argument("--train_planner_name", required=True, type=str)
    parser.add_argument("--test_planner_name", required=True, type=str)
    parser.add_argument("--planner_type", required=True, type=str,
                        choices=["pure", "ploi", "cmpl", "relx", "flax"])
    parser.add_argument("--num_seeds", required=True, type=int)
    parser.add_argument("--num_train_problems", required=True, type=int)
    parser.add_argument("--num_test_problems", required=True, type=int)
    parser.add_argument("--train_timeout", type=float, default=120.0)
    parser.add_argument("--test_timeout", required=True, type=float)
    parser.add_argument("--num_epochs", type=int, default=301)
    parser.add_argument("--cmpl_rules", type=str, default="config/mazenamo_complementary_rules.json")
    parser.add_argument("--relx_rules", type=str, default="config/mazenamo_relaxation_rules.json")
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--out", type=str, default="results/bce_vs_mus_results.json")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bce = _run_one(repo_root, "gnn-bce-10", args)
    mus = _run_one(repo_root, "gnn-grape-must", args)

    winner, rationale = _winner_label(bce, mus)

    report = {
        "config": {
            "domain_name": args.domain_name,
            "train_planner_name": args.train_planner_name,
            "test_planner_name": args.test_planner_name,
            "planner_type": args.planner_type,
            "num_seeds": args.num_seeds,
            "num_train_problems": args.num_train_problems,
            "num_test_problems": args.num_test_problems,
            "train_timeout": args.train_timeout,
            "test_timeout": args.test_timeout,
            "num_epochs": args.num_epochs,
            "cmpl_rules": args.cmpl_rules,
            "relx_rules": args.relx_rules,
            "repo_root": str(repo_root),
        },
        "bce": asdict(bce),
        "mus": asdict(mus),
        "comparison": {
            "winner": winner,
            "rationale": rationale,
            "primary_metric": "success_rate",
            "secondary_metric": "avg_planning_time",
            "tertiary_metric": "avg_plan_length",
            "deltas": {
                "mus_minus_bce_success_rate": (
                    None if (mus.avg_success_rate is None or bce.avg_success_rate is None)
                    else mus.avg_success_rate - bce.avg_success_rate
                ),
                "mus_minus_bce_planning_time": (
                    None if (mus.avg_planning_time is None or bce.avg_planning_time is None)
                    else mus.avg_planning_time - bce.avg_planning_time
                ),
                "mus_minus_bce_plan_length": (
                    None if (mus.avg_plan_length is None or bce.avg_plan_length is None)
                    else mus.avg_plan_length - bce.avg_plan_length
                ),
            },
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Summary ===")
    print(
        f"BCE success rate:   {_format_metric(bce.avg_success_rate, bce.success_rate_list)}\n"
        f"MUS success rate:   {_format_metric(mus.avg_success_rate, mus.success_rate_list)}"
    )
    print(
        f"BCE planning time:  {_format_metric(bce.avg_planning_time, bce.planning_time_list)}\n"
        f"MUS planning time:  {_format_metric(mus.avg_planning_time, mus.planning_time_list)}"
    )
    print(
        f"BCE plan length:    {_format_metric(bce.avg_plan_length, bce.plan_length_list)}\n"
        f"MUS plan length:    {_format_metric(mus.avg_plan_length, mus.plan_length_list)}"
    )
    print(f"Winner: {winner}")
    print(f"Reason: {rationale}")
    print(f"Saved report to: {out_path}")

    if bce.returncode != 0 or mus.returncode != 0:
        print("\nAt least one run failed. Check the saved stdout/stderr in the JSON report.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())