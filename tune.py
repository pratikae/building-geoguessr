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
  parser.add_argument("--checkpoint_dir", default="tuning_checkpoints", help="Directory to store per-trial checkpoints")
  parser.add_argument("--logs_dir", default="tuning_logs", help="Directory to store per-trial stdout/stderr logs")
  parser.add_argument("--python", default=sys.executable, help="Python interpreter to run train.py")
  parser.add_argument("--seed", type=int, default=493, help="Random seed for deterministic random search")
  parser.add_argument("--rerun_errors", action="store_true", help="Rerun trials that previously failed")
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

  # deterministic random search.
  rng = random.Random(args.seed)
  rng.shuffle(grid)

  return grid[: args.n_trials]


def get_header():
  return [
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
    "checkpoint_dir",
    "metrics_path",
    "stdout_path",
    "stderr_path",
    "stdout",
    "stderr",
  ]


def load_existing_results(output_path):
  output_path = Path(output_path)
  if not output_path.exists():
    return []

  with open(output_path, "r", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    return list(reader)


def save_results(output_path, rows):
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  header = get_header()

  # Atomic-ish write: write temp file, then replace real CSV.
  tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

  with open(tmp_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()
    for row in rows:
      writer.writerow({k: row.get(k, "") for k in header})

  tmp_path.replace(output_path)


def completed_trial_statuses(rows):
  result = {}
  for row in rows:
    try:
      trial = int(row["trial"])
    except Exception:
      continue

    result[trial] = row.get("status", "")

  return result


def should_skip_trial(existing_statuses, trial_idx, args):
  status = existing_statuses.get(trial_idx)

  if status == "ok":
    return True

  if status == "error" and not args.rerun_errors:
    return True

  # "running" or missing attempt should be rerun.
  return False


def write_running_row(args, trial_idx, params, metrics_path, trial_checkpoint_dir, stdout_path, stderr_path):
  return {
    "trial": trial_idx,
    "status": "running",
    "checkpoint_dir": str(trial_checkpoint_dir),
    "metrics_path": str(metrics_path),
    "stdout_path": str(stdout_path),
    "stderr_path": str(stderr_path),
    **params,
  }


def run_trial(args, trial_idx, params):
  metrics_dir = Path(args.metrics_dir)
  checkpoints_root = Path(args.checkpoint_dir)
  logs_dir = Path(args.logs_dir)

  metrics_dir.mkdir(parents=True, exist_ok=True)
  checkpoints_root.mkdir(parents=True, exist_ok=True)
  logs_dir.mkdir(parents=True, exist_ok=True)

  metrics_path = metrics_dir / f"trial_{trial_idx:03d}.json"
  trial_checkpoint_dir = checkpoints_root / f"trial_{trial_idx:03d}"
  trial_checkpoint_dir.mkdir(parents=True, exist_ok=True)

  stdout_path = logs_dir / f"trial_{trial_idx:03d}.out"
  stderr_path = logs_dir / f"trial_{trial_idx:03d}.err"

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

    # This is the important part for model-level checkpointing.
    # train.py needs to save checkpoints here so if/when it gets killed by hyak it can still resume.
    "--checkpoint_dir",
    str(trial_checkpoint_dir),
  ]

  if args.epochs is not None:
    command += ["--epochs", str(args.epochs)]

  print("Running trial", trial_idx, params)
  print(" ", " ".join(command))
  print(" checkpoints:", trial_checkpoint_dir)
  print(" metrics:", metrics_path)
  print(" stdout:", stdout_path)
  print(" stderr:", stderr_path)

  if args.dry_run:
    return None

  with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
    res = subprocess.run(
      command,
      stdout=stdout_file,
      stderr=stderr_file,
      text=True,
    )

  stdout_text = safe_read_text(stdout_path)
  stderr_text = safe_read_text(stderr_path)

  if res.returncode != 0:
    print("Trial failed:", trial_idx)
    return {
      "status": "error",
      "stdout": tail_text(stdout_text),
      "stderr": tail_text(stderr_text),
      "checkpoint_dir": str(trial_checkpoint_dir),
      "metrics_path": str(metrics_path),
      "stdout_path": str(stdout_path),
      "stderr_path": str(stderr_path),
      **params,
    }

  try:
    metrics = json.loads(metrics_path.read_text())
  except Exception as exc:
    print("Unable to read metrics file:", exc)
    return {
      "status": "error",
      "stdout": tail_text(stdout_text),
      "stderr": tail_text(stderr_text) + f"\nUnable to read metrics file: {exc}",
      "checkpoint_dir": str(trial_checkpoint_dir),
      "metrics_path": str(metrics_path),
      "stdout_path": str(stdout_path),
      "stderr_path": str(stderr_path),
      **params,
    }

  return {
    "status": "ok",
    "stdout": tail_text(stdout_text),
    "stderr": tail_text(stderr_text),
    "checkpoint_dir": str(trial_checkpoint_dir),
    "metrics_path": str(metrics_path),
    "stdout_path": str(stdout_path),
    "stderr_path": str(stderr_path),
    **params,
    **metrics,
  }


def safe_read_text(path):
  try:
    return Path(path).read_text(errors="replace")
  except Exception:
    return ""


def tail_text(text, max_chars=4000):
  if len(text) <= max_chars:
    return text
  return text[-max_chars:]


def replace_or_append_row(rows, new_row):
  trial = int(new_row["trial"])

  new_rows = []
  replaced = False

  for row in rows:
    try:
      row_trial = int(row["trial"])
    except Exception:
      new_rows.append(row)
      continue

    if row_trial == trial:
      new_rows.append(new_row)
      replaced = True
    else:
      new_rows.append(row)

  if not replaced:
    new_rows.append(new_row)

  new_rows.sort(key=lambda r: int(r.get("trial", 0)))
  return new_rows


def main():
  args = parse_args()

  if args.search == "grid":
    trials = build_grid(args)
  else:
    trials = sample_random(args)

  rows = load_existing_results(args.output)
  existing_statuses = completed_trial_statuses(rows)

  print(f"Loaded {len(rows)} existing result rows from {args.output}")

  for idx, params in enumerate(trials, start=1):
    if should_skip_trial(existing_statuses, idx, args):
      print(f"Skipping trial {idx}, already has status={existing_statuses[idx]}")
      continue

    metrics_path = Path(args.metrics_dir) / f"trial_{idx:03d}.json"
    trial_checkpoint_dir = Path(args.checkpoint_dir) / f"trial_{idx:03d}"
    stdout_path = Path(args.logs_dir) / f"trial_{idx:03d}.out"
    stderr_path = Path(args.logs_dir) / f"trial_{idx:03d}.err"

    running_row = write_running_row(
      args,
      idx,
      params,
      metrics_path,
      trial_checkpoint_dir,
      stdout_path,
      stderr_path,
    )

    rows = replace_or_append_row(rows, running_row)
    save_results(args.output, rows)

    row = run_trial(args, idx, params)

    if row is not None:
      row["trial"] = idx
      rows = replace_or_append_row(rows, row)
      save_results(args.output, rows)
      existing_statuses[idx] = row["status"]

  print("tuning done, results saved to", args.output)


if __name__ == "__main__":
  main()