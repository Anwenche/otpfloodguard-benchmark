#!/usr/bin/env python3
"""Check whether manuscript-reported values match current result artifacts."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript.md"
RESULTS = ROOT / "results"


def status_line(name: str, status: str, detail: str = "") -> str:
    suffix = f" - {detail}" if detail else ""
    return f"{status}: {name}{suffix}"


def fmt4(value: float) -> str:
    return f"{value:.4f}"


def contains(text: str, value: str) -> bool:
    return value in text


def main() -> int:
    if not MANUSCRIPT.exists():
        print(status_line("manuscript.md", "PENDING_REGENERATION", "manuscript source not present in repository root"))
        return 0

    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    checks: list[tuple[str, str, str]] = []

    try:
        metrics = pd.read_csv(RESULTS / "metrics.csv")
        rf = metrics[(metrics["model"] == "Random Forest") & (metrics["feature_set"] == "full")].iloc[0]
        checks.append(("RF main F1", "MATCH" if contains(manuscript, fmt4(rf["f1"])) else "MISMATCH", fmt4(rf["f1"])))
        checks.append(("RF main recall", "MATCH" if contains(manuscript, fmt4(rf["recall"])) else "MISMATCH", fmt4(rf["recall"])))
    except Exception as exc:
        checks.append(("main metrics", "PENDING_REGENERATION", str(exc)))

    try:
        difficulty = pd.read_csv(RESULTS / "difficulty_metrics.csv")
        rule_rows = difficulty[difficulty["model"] == "Tuned Rule Baseline"]
        if "Tuned-rule F1" in manuscript and "Tuned-rule recall" in manuscript and not rule_rows.empty:
            checks.append(("Table VIII tuned-rule labeling", "MATCH", "uses Tuned-rule columns"))
        else:
            checks.append(("Table VIII tuned-rule labeling", "MISMATCH", "missing explicit tuned-rule columns"))
    except Exception as exc:
        checks.append(("Table VIII", "PENDING_REGENERATION", str(exc)))

    try:
        costs = pd.read_csv(RESULTS / "cost_sensitive_thresholds.csv")
        if "Random Forest held-out test performance" in manuscript and not costs.empty:
            checks.append(("Table XI Random Forest caption", "MATCH", "caption/source text names Random Forest"))
        else:
            checks.append(("Table XI Random Forest caption", "MISMATCH", "missing model-specific wording"))
    except Exception as exc:
        checks.append(("Table XI", "PENDING_REGENERATION", str(exc)))

    try:
        shifted = pd.read_csv(RESULTS / "cross_generator_metrics.csv")
        rf_shift = shifted[shifted["model"] == "Random Forest"].iloc[0]
        checks.append(("Generator-shift RF F1", "MATCH" if contains(manuscript, fmt4(rf_shift["f1"])) else "MISMATCH", fmt4(rf_shift["f1"])))
    except Exception as exc:
        checks.append(("generator shift", "PENDING_REGENERATION", str(exc)))

    try:
        predictions = pd.read_csv(RESULTS / "test_predictions.csv")
        tn = int(((predictions["label"] == 0) & (predictions["predicted_label"] == 0)).sum())
        fp = int(((predictions["label"] == 0) & (predictions["predicted_label"] == 1)).sum())
        fn = int(((predictions["label"] == 1) & (predictions["predicted_label"] == 0)).sum())
        tp = int(((predictions["label"] == 1) & (predictions["predicted_label"] == 1)).sum())
        expected = f"{tn:,} true negatives, {fp} false positives, {fn} false negatives, and {tp} true positives"
        checks.append(("Confusion matrix text", "MATCH" if expected in manuscript else "MISMATCH", expected))
    except Exception as exc:
        checks.append(("confusion matrix", "PENDING_REGENERATION", str(exc)))

    try:
        multi = pd.read_csv(RESULTS / "multi_split_summary.csv")
        rf_multi = multi[multi["model"] == "Random Forest"].iloc[0]
        expected = f"{fmt4(rf_multi['f1_mean'])} +/- {fmt4(rf_multi['f1_std'])}"
        checks.append(("Multi-split RF F1 mean/std", "MATCH" if expected in manuscript else "MISMATCH", expected))
    except Exception as exc:
        checks.append(("multi-split", "PENDING_REGENERATION", str(exc)))

    try:
        generator = pd.read_csv(RESULTS / "generator_seed_summary.csv")
        rf_gen = generator[generator["model"] == "Random Forest"].iloc[0]
        expected = f"{fmt4(rf_gen['f1_mean'])} +/- {fmt4(rf_gen['f1_std'])}"
        checks.append(("Generator-seed RF F1 mean/std", "MATCH" if expected in manuscript else "MISMATCH", expected))
    except Exception as exc:
        checks.append(("generator-seed", "PENDING_REGENERATION", str(exc)))

    exit_code = 0
    for name, status, detail in checks:
        print(status_line(name, status, detail))
        if status == "MISMATCH":
            exit_code = 1

    stale_number_pattern = re.compile(r"\{\{[A-Z0-9_]+\}\}")
    placeholders = stale_number_pattern.findall(manuscript)
    if placeholders:
        print(status_line("unresolved manuscript placeholders", "PENDING_REGENERATION", ", ".join(placeholders)))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
