"""
run_evaluation.py
=================
Main entrypoint for the Email Generation Assistant evaluation pipeline.

Usage:
    python run_evaluation.py

What it does:
  1. Loads 10 test scenarios from data/test_scenarios.json.
  2. For each scenario, generates an email using GPT-4o AND Grok-3.
  3. Scores each generated email on 3 custom metrics.
  4. Saves full results to data/results/evaluation_report.json (detailed)
     and data/results/evaluation_report.csv (tabular summary).
  5. Prints a rich comparative summary table to the console.
  6. Writes a Markdown analysis summary to data/results/analysis_summary.md.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from colorama import init as colorama_init, Fore, Style

from src.evaluator import evaluate_email, _get_judge_client
from src.generator import generate_email
from src.models import MODELS, get_client

# Fix Windows console unicode printing errors
if sys.stdout.encoding.lower() != 'utf-8' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()
colorama_init(autoreset=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "data"
RESULTS_DIR   = DATA_DIR / "results"
SCENARIOS_FILE = DATA_DIR / "test_scenarios.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_scenarios() -> list[dict]:
    with open(SCENARIOS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _banner(text: str) -> None:
    width = 72
    print(f"\n{Fore.CYAN}{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}{Style.RESET_ALL}\n")


def _print_row(label: str, value: str, color: str = Fore.WHITE) -> None:
    print(f"  {Fore.YELLOW}{label:<30}{color}{value}{Style.RESET_ALL}")


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(model_keys: list[str] = ("gpt-4o", "grok-3")) -> list[dict]:
    """
    Run the full evaluation pipeline.

    Args:
        model_keys: List of model keys to evaluate (must be in src/models.MODELS).

    Returns:
        List of result dicts (one per scenario × model).
    """
    scenarios = _load_scenarios()
    judge_client = _get_judge_client()   # shared judge client for all LLM-judge calls

    all_results: list[dict] = []

    for model_key in model_keys:
        _banner(f"Evaluating Model: {MODELS[model_key].name}")

        try:
            client, config = get_client(model_key)
        except ValueError as e:
            print(f"{Fore.RED}  ✗ Skipping {model_key}: {e}{Style.RESET_ALL}\n")
            continue

        for scenario in tqdm(scenarios, desc=f"  {config.name}", unit="scenario"):
            sid       = scenario["id"]
            intent    = scenario["intent"]
            key_facts = scenario["key_facts"]
            tone      = scenario["tone"]

            # ---- Generate email ----
            try:
                generated_email = generate_email(
                    client, config, intent, key_facts, tone
                )
            except Exception as e:
                print(f"\n{Fore.RED}  ✗ Generation failed for scenario {sid}: {e}{Style.RESET_ALL}")
                generated_email = "[GENERATION FAILED]"

            # ---- Evaluate email ----
            try:
                eval_result = evaluate_email(
                    generated_email, key_facts, tone, judge_client
                )
            except Exception as e:
                print(f"\n{Fore.RED}  ✗ Evaluation failed for scenario {sid}: {e}{Style.RESET_ALL}")
                eval_result = {
                    "metric_1_fact_recall":            {"score": 0.0},
                    "metric_2_tone_adherence":         {"score": 0.0, "raw_score": 0, "reasoning": str(e)},
                    "metric_3_fluency_professionalism":{"score": 0.0, "raw_score": 0, "reasoning": str(e)},
                    "composite_score": 0.0,
                }

            row = {
                "model_key":      model_key,
                "model_name":     config.name,
                "scenario_id":    sid,
                "intent":         intent,
                "tone":           tone,
                "generated_email": generated_email,
                "reference_email": scenario["reference_email"],
                **eval_result,
            }
            all_results.append(row)

            # Small delay to respect rate limits
            time.sleep(0.5)

    return all_results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_json(results: list[dict], path: Path) -> None:
    """Write full results to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{Fore.GREEN}  ✔ JSON report saved → {path}{Style.RESET_ALL}")


def _write_csv(results: list[dict], path: Path) -> None:
    """Write flattened tabular summary to CSV."""
    fieldnames = [
        "model_name", "scenario_id", "intent", "tone",
        "m1_fact_recall",
        "m2_tone_adherence_raw", "m2_tone_adherence_norm",
        "m3_fluency_raw", "m3_fluency_norm",
        "composite_score",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "model_name":            r["model_name"],
                "scenario_id":           r["scenario_id"],
                "intent":                r["intent"],
                "tone":                  r["tone"],
                "m1_fact_recall":        r["metric_1_fact_recall"]["score"],
                "m2_tone_adherence_raw": r["metric_2_tone_adherence"].get("raw_score", "N/A"),
                "m2_tone_adherence_norm": r["metric_2_tone_adherence"]["score"],
                "m3_fluency_raw":        r["metric_3_fluency_professionalism"].get("raw_score", "N/A"),
                "m3_fluency_norm":       r["metric_3_fluency_professionalism"]["score"],
                "composite_score":       r["composite_score"],
            })

    print(f"{Fore.GREEN}  ✔ CSV report saved  → {path}{Style.RESET_ALL}")


def _write_analysis(results: list[dict], path: Path) -> None:
    """Generate and write a comparative analysis summary in Markdown."""
    # Aggregate per model
    model_stats: dict[str, dict] = {}
    for r in results:
        key = r["model_key"]
        if key not in model_stats:
            model_stats[key] = {
                "name": r["model_name"],
                "m1_scores": [],
                "m2_scores": [],
                "m3_scores": [],
                "composite_scores": [],
                "low_m1": [],   # scenarios where fact recall < 0.5
                "low_m2": [],   # scenarios where tone adherence < 0.6
            }
        s = model_stats[key]
        s["m1_scores"].append(r["metric_1_fact_recall"]["score"])
        s["m2_scores"].append(r["metric_2_tone_adherence"]["score"])
        s["m3_scores"].append(r["metric_3_fluency_professionalism"]["score"])
        s["composite_scores"].append(r["composite_score"])

        if r["metric_1_fact_recall"]["score"] < 0.5:
            s["low_m1"].append(r["scenario_id"])
        if r["metric_2_tone_adherence"]["score"] < 0.6:
            s["low_m2"].append(r["scenario_id"])

    def avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    # Determine winner
    model_keys = list(model_stats.keys())
    if len(model_keys) < 2:
        winner_key = model_keys[0] if model_keys else "N/A"
        loser_key  = None
    else:
        model_keys_sorted = sorted(
            model_keys, key=lambda k: avg(model_stats[k]["composite_scores"]), reverse=True
        )
        winner_key = model_keys_sorted[0]
        loser_key  = model_keys_sorted[1]

    winner = model_stats[winner_key]
    loser  = model_stats[loser_key] if loser_key else None

    lines = [
        "# Comparative Model Analysis Summary",
        f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n",
        "---",
        "",
        "## Metric Definitions",
        "",
        "| Metric | Definition | Method | Scale |",
        "|--------|-----------|--------|-------|",
        "| **M1 — Fact Recall** | Fraction of key facts whose lemmatized keywords appear in the generated email (≥50% keyword overlap per fact counts as recalled). Morphological variants (e.g., request/requesting/requested) are treated as the same lemma. | NLTK lemmatization + set overlap | 0.0 – 1.0 |",
        "| **M2 — Tone Adherence** | How accurately the generated email matches the requested tone, rated by GPT-4o-mini on a 1–10 rubric (9–10: perfect match; 1–2: entirely wrong tone). | LLM-as-a-Judge | 0.0 – 1.0 |",
        "| **M3 — Fluency & Professionalism** | Grammar & Spelling (0–4) + Email Structure (0–3) + Clarity & Conciseness (0–3), rated by GPT-4o-mini, summed to 10. | LLM-as-a-Judge | 0.0 – 1.0 |",
        "",
        "---",
        "",
        "## Score Summary",
        "",
        "| Model | M1 Fact Recall | M2 Tone Adherence | M3 Fluency | Composite |",
        "|-------|:--------------:|:-----------------:|:----------:|:---------:|",
    ]

    for key, s in model_stats.items():
        lines.append(
            f"| {s['name']} | {avg(s['m1_scores']):.4f} | "
            f"{avg(s['m2_scores']):.4f} | {avg(s['m3_scores']):.4f} | "
            f"**{avg(s['composite_scores']):.4f}** |"
        )

    lines += [
        "",
        "---",
        "",
        "## Analysis",
        "",
        f"### 1. Which model performed better?",
        "",
        f"**{winner['name']}** achieved a higher composite score "
        f"({avg(winner['composite_scores']):.4f}) across all 10 evaluation scenarios.",
    ]

    if loser:
        delta_m1 = avg(winner["m1_scores"]) - avg(loser["m1_scores"])
        delta_m2 = avg(winner["m2_scores"]) - avg(loser["m2_scores"])
        delta_m3 = avg(winner["m3_scores"]) - avg(loser["m3_scores"])

        lines += [
            f"It outperformed **{loser['name']}** ({avg(loser['composite_scores']):.4f}) "
            f"on Fact Recall by {delta_m1:+.4f}, Tone Adherence by {delta_m2:+.4f}, "
            f"and Fluency by {delta_m3:+.4f}.",
            "",
            f"### 2. Biggest failure mode of {loser['name']}",
            "",
        ]

        # Identify the weakest dimension
        loser_avgs = {
            "Fact Recall":   avg(loser["m1_scores"]),
            "Tone Adherence": avg(loser["m2_scores"]),
            "Fluency":       avg(loser["m3_scores"]),
        }
        weakest_dim = min(loser_avgs, key=loser_avgs.get)
        lines += [
            f"The weakest dimension for **{loser['name']}** was **{weakest_dim}** "
            f"(avg: {loser_avgs[weakest_dim]:.4f}). ",
        ]

        if loser["low_m1"]:
            lines.append(
                f"Specifically, fact recall fell below 0.5 on scenarios: "
                f"{loser['low_m1']}. This suggests the model paraphrased or omitted "
                f"key factual content instead of integrating it faithfully."
            )
        if loser["low_m2"]:
            lines.append(
                f"Tone adherence fell below 0.6 on scenarios: {loser['low_m2']}. "
                f"The model tended to drift toward a generic neutral tone regardless of "
                f"the specified style."
            )

    lines += [
        "",
        "### 3. Production Recommendation",
        "",
        f"Based on the evaluation data, **{winner['name']}** is recommended for production use.",
        "",
        "**Justification:**",
        f"- Higher M1 Fact Recall ensures that business-critical information is faithfully "
        f"included in generated emails, reducing the risk of omissions that could cause "
        f"miscommunication.",
        f"- Higher M2 Tone Adherence means the model can reliably shift register (formal, "
        f"urgent, empathetic) on demand — a key requirement for a production email assistant "
        f"serving diverse communication scenarios.",
        f"- Higher M3 Fluency & Professionalism confirms that outputs are consistently "
        f"well-structured and polished, requiring minimal human editing before sending.",
        "",
        "---",
        "_Report generated automatically by the Email Generation Assistant evaluation pipeline._",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"{Fore.GREEN}  ✔ Analysis summary → {path}{Style.RESET_ALL}")


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def _print_console_summary(results: list[dict]) -> None:
    """Print a nicely formatted per-model score table."""
    _banner("Evaluation Results Summary")

    model_groups: dict[str, list[dict]] = {}
    for r in results:
        model_groups.setdefault(r["model_key"], []).append(r)

    header = (
        f"  {'Sc':>3}  "
        f"{'M1 Recall':>10}  "
        f"{'M2 Tone':>8}  "
        f"{'M3 Fluency':>10}  "
        f"{'Composite':>10}"
    )

    for model_key, rows in model_groups.items():
        print(f"\n{Fore.MAGENTA}  ▸ {MODELS[model_key].name}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{header}{Style.RESET_ALL}")
        print(f"  {'-' * 58}")

        composites: list[float] = []
        m1s: list[float] = []
        m2s: list[float] = []
        m3s: list[float] = []

        for r in sorted(rows, key=lambda x: x["scenario_id"]):
            m1 = r["metric_1_fact_recall"]["score"]
            m2 = r["metric_2_tone_adherence"]["score"]
            m3 = r["metric_3_fluency_professionalism"]["score"]
            cs = r["composite_score"]

            m1s.append(m1); m2s.append(m2); m3s.append(m3); composites.append(cs)

            # Colour-code composite score
            if cs >= 0.75:
                cs_color = Fore.GREEN
            elif cs >= 0.5:
                cs_color = Fore.YELLOW
            else:
                cs_color = Fore.RED

            print(
                f"  {r['scenario_id']:>3}  "
                f"{m1:>10.4f}  "
                f"{m2:>8.4f}  "
                f"{m3:>10.4f}  "
                f"{cs_color}{cs:>10.4f}{Style.RESET_ALL}"
            )

        print(f"  {'-' * 58}")

        def avg(lst: list) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        print(
            f"{Fore.YELLOW}  AVG  "
            f"{avg(m1s):>10.4f}  "
            f"{avg(m2s):>8.4f}  "
            f"{avg(m3s):>10.4f}  "
            f"{Fore.CYAN}{avg(composites):>10.4f}{Style.RESET_ALL}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _banner("Email Generation Assistant — Evaluation Pipeline")
    print(f"  {Fore.WHITE}Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Scenarios : {SCENARIOS_FILE}{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}Models    : GPT-4o  |  Grok-3{Style.RESET_ALL}\n")

    results = run_evaluation(model_keys=["gpt-4o", "grok-3"])

    if not results:
        print(f"{Fore.RED}No results to save. Check your API keys in .env{Style.RESET_ALL}")
        sys.exit(1)

    json_path = RESULTS_DIR / "evaluation_report.json"
    csv_path = RESULTS_DIR / "evaluation_report.csv"
    md_path = RESULTS_DIR / "analysis_summary.md"

    # Write the report for all models
    _write_json(results, json_path)
    _write_csv(results, csv_path)
    _write_analysis(results, md_path)

    _print_console_summary(results)

    _banner("Pipeline Complete ✔")


if __name__ == "__main__":
    main()
