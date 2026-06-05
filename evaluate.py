import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import (
    HuggingFaceImageDataset,
    build_label_mappings,
    collate_fn,
    default_transforms,
    ensure_dataset,
    resolve_image_column,
)
from model import BobTheBuilder


class EvalHuggingFaceImageDataset(HuggingFaceImageDataset):
    def __getitem__(self, idx):
        image, targets = super().__getitem__(idx)
        sample = self.dataset[idx]
        targets["year"] = int(sample["Year"])
        return image, targets


def eval_collate_fn(batch):
    images, targets_list = zip(*batch)
    images = torch.stack(images)

    collated: Dict[str, torch.Tensor] = {}
    for key in targets_list[0].keys():
        if key == "year":
            collated[key] = torch.tensor([t[key] for t in targets_list], dtype=torch.long)
        else:
            collated[key] = torch.tensor([t[key] for t in targets_list], dtype=torch.long)
    return images, collated


def build_dataloader(config: Dict[str, Any], split: str, batch_size: int):
    ds = ensure_dataset(config)
    if split not in ds:
        raise ValueError(f"Unknown split '{split}'. Available splits: {list(ds.keys())}")

    image_column = resolve_image_column(ds[split])
    mappings = build_label_mappings(ds["train"])
    transforms = default_transforms(config.get("data", {}).get("image_size", 224))

    eval_ds = EvalHuggingFaceImageDataset(ds[split], mappings, image_column, transforms)
    return DataLoader(eval_ds, batch_size=batch_size, shuffle=False, collate_fn=eval_collate_fn)


def load_model(checkpoint_path: Path, num_countries: int, num_us_states: int, config: Dict[str, Any], device: torch.device):
    model = BobTheBuilder(num_countries, num_us_states, config)
    model.to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def evaluate(model: torch.nn.Module, dataloader: DataLoader):
    device = next(model.parameters()).device
    total = 0
    correct_country = 0
    total_us = 0
    correct_us = 0
    decade_stats: Dict[int, Dict[str, int]] = {}

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)

            country_preds = torch.argmax(outputs["country"], dim=1).cpu()
            us_preds = torch.argmax(outputs["us_state"], dim=1).cpu()
            country_labels = targets["country"]
            us_labels = targets["us_state"]
            is_us = targets.get("is_us", torch.zeros_like(country_labels))
            years = targets["year"]

            total += country_labels.size(0)
            correct_country += (country_preds == country_labels).sum().item()

            mask_us = is_us == 1
            if mask_us.any():
                total_us += mask_us.sum().item()
                correct_us += (us_preds[mask_us] == us_labels[mask_us]).sum().item()

            for year, pred, label in zip(years.tolist(), country_preds.tolist(), country_labels.tolist()):
                decade = year - (year % 10)
                bucket = decade_stats.setdefault(decade, {"total": 0, "correct": 0})
                bucket["total"] += 1
                if pred == label:
                    bucket["correct"] += 1

    results = {
        "split_total": total,
        "country_accuracy": 100.0 * correct_country / total if total else 0.0,
        "us_state_accuracy": 100.0 * correct_us / total_us if total_us else 0.0,
        "decade_accuracy": {
            str(decade): 100.0 * stats["correct"] / stats["total"]
            for decade, stats in sorted(decade_stats.items())
        },
        "decade_counts": {
            str(decade): stats["total"]
            for decade, stats in sorted(decade_stats.items())
        },
    }
    return results


def plot_results(results: Dict[str, Any], plots_dir: Path, split: str):
    plots_dir.mkdir(parents=True, exist_ok=True)

    # decade accuracy bar chart
    decade_acc = results["decade_accuracy"]
    decade_counts = results.get("decade_counts", {})
    if decade_acc:
        decades = [int(d) for d in decade_acc.keys()]
        accs = [decade_acc[str(d)] for d in decades]
        counts = [decade_counts.get(str(d), 0) for d in decades]
        labels = [f"{d}s" for d in decades]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(labels, accs, color="steelblue", edgecolor="white")
        ax.set_xlabel("Decade")
        ax.set_ylabel("Country Accuracy (%)")
        ax.set_title(f"Country Accuracy by Decade ({split} split)")
        ax.set_ylim(0, 110)
        ax.axhline(results["country_accuracy"], color="tomato", linestyle="--", linewidth=1.5, label=f"Overall ({results['country_accuracy']:.1f}%)")
        ax.legend()
        for bar, val, n in zip(bars, accs, counts):
            x = bar.get_x() + bar.get_width() / 2
            ax.text(x, bar.get_height() + 1, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
            ax.text(x, bar.get_height() + 5.5, f"n={n}", ha="center", va="bottom", fontsize=7, color="gray")
        fig.tight_layout()
        fig.savefig(plots_dir / f"{split}_decade_accuracy.png", dpi=150)
        plt.close(fig)

    # overall accuracy summary bar chart
    summary_labels = ["Country", "US State"]
    summary_vals = [results["country_accuracy"], results["us_state_accuracy"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(summary_labels, summary_vals, color=["steelblue", "seagreen"], edgecolor="white")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Overall Accuracy ({split} split)")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, summary_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{split}_overall_accuracy.png", dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset split.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint file")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--model_name", default=None, help="Name tag used to organise outputs into a subdirectory")
    parser.add_argument("--metrics_path", default=None, help="Optional JSON path to save metrics")
    parser.add_argument("--plots_dir", default=None, help="Optional directory to save accuracy plots")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ensure_dataset(config)
    num_countries = len(build_label_mappings(ds["train"])["country"]["mapping"])
    num_us_states = len(build_label_mappings(ds["train"])["us_state"]["mapping"])

    model = load_model(Path(args.checkpoint), num_countries, num_us_states, config, device)
    dataloader = build_dataloader(config, args.split, args.batch_size)

    results = evaluate(model, dataloader)
    print(json.dumps(results, indent=2))

    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        if args.model_name:
            metrics_path = metrics_path.parent / args.model_name / metrics_path.name
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as fp:
            json.dump(results, fp, indent=2)

    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
        if args.model_name:
            plots_dir = plots_dir / args.model_name
        plot_results(results, plots_dir, args.split)
        print(f"Plots saved to {plots_dir}")


if __name__ == "__main__":
    main()
