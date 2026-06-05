#!/usr/bin/env python3
"""Find the best finetune and linear_probe models across all tuning result CSVs."""

import csv
import sys
from pathlib import Path

RESULTS_DIR = Path("tune_results")


def load_all_results():
    rows = []
    for csv_path in RESULTS_DIR.glob("tuning_results_*.csv"):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok" and row.get("best_val_acc"):
                    row["_source"] = csv_path.name
                    rows.append(row)
    return rows


def best_for_type(rows, experiment_type):
    candidates = [r for r in rows if r.get("experiment_type") == experiment_type]
    if not candidates:
        return None
    return max(candidates, key=lambda r: float(r["best_val_acc"]))


def print_winner(label, row):
    if row is None:
        print(f"{label}: no completed runs found")
        return
    print(f"{label}:")
    print(f"  best_val_acc  : {float(row['best_val_acc']):.4f}")
    print(f"  lr            : {row['lr']}")
    print(f"  batch_size    : {row['batch_size']}")
    print(f"  weight_decay  : {row['weight_decay']}")
    print(f"  checkpoint_dir: {row['checkpoint_dir']}")
    print(f"  source        : {row['_source']}")


def main():
    rows = load_all_results()
    if not rows:
        print(f"No completed results found in {RESULTS_DIR}/", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(rows)} completed trial(s) across all runs.\n")
    print_winner("Best finetune (ft)", best_for_type(rows, "finetune"))
    print()
    print_winner("Best linear_probe (lp)", best_for_type(rows, "linear_probe"))


if __name__ == "__main__":
    main()
