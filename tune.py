import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

def parse_args():
  parser = argparse.ArgumentParser(description="Hyperparameter tuning for BobTheBuilder")
  parser.add_argument("--config", default="config.yaml", help="Base config file path")
  parser.add_argument("--output", default="tuning_results.csv", help="CSV output file")
  parser.add_argument("--search", choices=["grid", "random"], default="grid", help="Search strategy")
  parser.add_argument("--n_trials", type=int, default=20, help="Number of trials for random search")
  parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
  parser.add_argument("--lrs", nargs="+", type=float, default=[1e-4, 3e-5, 1e-5], help="Learning rates to try")
  parser.add_argument("--batch_sizes", nargs="+", type=int, default=[32, 64], help="Batch sizes to try")
  parser.add_argument("--weight_decays", nargs="+", type=float, default=[0.0, 1e-2], help="Weight decay values to try")
  parser.add_argument("--experiment_types", nargs="+", default=["linear_probe", "finetune"], help="Experiment types to try")
  parser.add_argument("--epochs", type=int, default=None, help="Override training epochs for tuning runs")
  parser.add_argument("--metrics_dir", default="tuning_metrics", help="Directory to store per-run JSON metrics")
  parser.add_argument("--python", default=sys.executable, help="Python interpreter to run train.py")
  return parser.parse_args()

def build_grid(args):
  grid = []
  for exp_type in args.experiment_types:
    for lr in args.lrs:
      for batch_size in args.batch_sizes:
        for weight_decay in args.weight_decays:
          grid.append(
            {
              "experiment_type": exp_type,
              "lr": lr,
              "batch_size": batch_size,
              "weight_decay": weight_decay,
            }
          )
  return grid


def sample_random(args):
  grid = build_grid(args)
  random.shuffle(grid)
  return grid[: args.n_trials]


def run_trial(args, trial_idx, params):
  metrics_dir = Path(args.metrics_dir)
  metrics_dir.mkdir(parents=True, exist_ok=True)
  metrics_path = metrics_dir / f"trial_{trial_idx:03d}.json"

  command = [
    args.python,
    "train.py",
    "--config",
    args.config,
    "--lr",
    str(params["lr"]),
    "--batch_size",
    str(params["batch_size"]),
    "--weight_decay",
    str(params["weight_decay"]),
    "--experiment_type",
    params["experiment_type"],
    "--metrics_path",
    str(metrics_path),
  ]

  if args.epochs is not None:
    command += ["--epochs", str(args.epochs)]

  print("Running trial", trial_idx, params)
  print(" ", " ".join(command))
  if args.dry_run:
    return None

  res = subprocess.run(command, capture_output=True, text=True)
  if res.returncode != 0:
    print("Trial failed:", res.stdout, res.stderr)
    return {
      "status": "error",
      "stdout": res.stdout,
      "stderr": res.stderr,
      **params,
    }

  try:
    metrics = json.loads(metrics_path.read_text())
  except Exception as exc:
    print("Unable to read metrics file:", exc)
    return {
      "status": "error",
      "stdout": res.stdout,
      "stderr": res.stderr,
      **params,
    }

  return {
    "status": "ok",
    "stdout": res.stdout,
    "stderr": res.stderr,
    **params,
    **metrics,
  }


def save_results(output_path, rows):
  header = [
    "trial",
    "experiment_type",
    "lr",
    "batch_size",
    "weight_decay",
    "status",
    "best_val_acc",
    "final_val_loss",
    "final_val_acc",
    "epochs",
    "stdout",
    "stderr",
  ]

  with open(output_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()
    for row in rows:
      writer.writerow({k: row.get(k, "") for k in header})


def main():
  args = parse_args()

  if args.search == "grid":
    trials = build_grid(args)
  else:
    trials = sample_random(args)

  rows = []
  for idx, params in enumerate(trials, start=1):
    row = run_trial(args, idx, params)
    if row is not None:
      row["trial"] = idx
      rows.append(row)
      save_results(args.output, rows)

  print("tuning done, results saved to", args.output)


if __name__ == "__main__":
    main()
