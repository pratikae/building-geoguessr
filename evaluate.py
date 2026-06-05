import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
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

try:
    import geopandas as gpd
    _HAS_GEOPANDAS = True
except ImportError:
    _HAS_GEOPANDAS = False


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


def evaluate(model: torch.nn.Module, dataloader: DataLoader, mappings: Dict[str, Any]):
    device = next(model.parameters()).device
    total = 0
    correct_country = 0
    total_us = 0
    correct_us = 0
    decade_stats: Dict[int, Dict] = {}
    per_country_stats: Dict[int, Dict[str, int]] = {}
    confusion: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    idx_to_country = {v: k for k, v in mappings["country"]["mapping"].items()}
    topk_ks = [1, 3, 5]

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)

            country_logits = outputs["country"]
            country_preds = torch.argmax(country_logits, dim=1).cpu()
            actual_topk = min(5, country_logits.size(1))
            topk_preds = torch.topk(country_logits, k=actual_topk, dim=1).indices.cpu()

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

            for year, pred, label, tk in zip(
                years.tolist(), country_preds.tolist(), country_labels.tolist(), topk_preds.tolist()
            ):
                decade = year - (year % 10)
                bucket = decade_stats.setdefault(decade, {"total": 0, "correct": 0, "topk": defaultdict(int)})
                bucket["total"] += 1
                if pred == label:
                    bucket["correct"] += 1
                for k in topk_ks:
                    if k <= actual_topk and label in tk[:k]:
                        bucket["topk"][k] += 1

                cs = per_country_stats.setdefault(label, {"total": 0, "correct": 0})
                cs["total"] += 1
                if pred == label:
                    cs["correct"] += 1

                confusion[label][pred] += 1

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
        "decade_topk_accuracy": {
            str(decade): {
                str(k): 100.0 * stats["topk"].get(k, 0) / stats["total"]
                for k in topk_ks if k <= actual_topk
            }
            for decade, stats in sorted(decade_stats.items())
        },
        "per_country_accuracy": {
            idx_to_country.get(idx, str(idx)): 100.0 * stats["correct"] / stats["total"]
            for idx, stats in sorted(per_country_stats.items())
        },
        "per_country_counts": {
            idx_to_country.get(idx, str(idx)): stats["total"]
            for idx, stats in sorted(per_country_stats.items())
        },
        "confusion": {
            idx_to_country.get(label, str(label)): {
                idx_to_country.get(pred, str(pred)): count
                for pred, count in preds.items()
            }
            for label, preds in confusion.items()
        },
    }
    return results


# --- plotting helpers ---------------------------------------------------------

def _trend_line(xs: List[int], ys: List[float]):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxy = sum(xi * yi for xi, yi in zip(xs, ys))
    sxx = sum(xi ** 2 for xi in xs)
    denom = n * sxx - sx ** 2
    if denom == 0:
        return [sy / n] * n, 0.0
    m = (n * sxy - sx * sy) / denom
    b = (sy - m * sx) / n
    return [m * xi + b for xi in xs], m


def _plot_decade_accuracy(results: Dict[str, Any], plots_dir: Path, split: str):
    decade_acc = results["decade_accuracy"]
    decade_counts = results.get("decade_counts", {})
    if not decade_acc:
        return

    decades = [int(d) for d in decade_acc.keys()]
    accs = [decade_acc[str(d)] for d in decades]
    counts = [decade_counts.get(str(d), 0) for d in decades]
    labels = [f"{d}s\n(n={n})" for d, n in zip(decades, counts)]
    xs = list(range(len(decades)))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, accs, color="steelblue", edgecolor="white")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Country Accuracy (%)")
    ax.set_title(f"Country Accuracy by Decade ({split} split)")
    ax.set_ylim(0, 115)

    if len(xs) >= 2:
        trend, m = _trend_line(xs, accs)
        direction = "improving" if m > 0 else "declining"
        ax.plot(xs, trend, color="tomato", linestyle="--", linewidth=1.5,
                label=f"Trend ({direction}, {m:+.1f}%/decade)")
        ax.legend()

    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(plots_dir / f"{split}_decade_accuracy.png", dpi=150)
    plt.close(fig)


def _plot_overall_accuracy(results: Dict[str, Any], plots_dir: Path, split: str):
    labels = ["Country", "US State"]
    vals = [results["country_accuracy"], results["us_state_accuracy"]]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, vals, color=["steelblue", "seagreen"], edgecolor="white")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Overall Accuracy ({split} split)")
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{split}_overall_accuracy.png", dpi=150)
    plt.close(fig)


def _plot_topk_decade(results: Dict[str, Any], plots_dir: Path, split: str):
    decade_topk = results.get("decade_topk_accuracy", {})
    decade_counts = results.get("decade_counts", {})
    if not decade_topk:
        return

    decades = sorted(int(d) for d in decade_topk.keys())
    ks = sorted(int(k) for k in next(iter(decade_topk.values())).keys())
    labels = [f"{d}s\n(n={decade_counts.get(str(d), 0)})" for d in decades]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue", "seagreen", "darkorange"]
    for i, k in enumerate(ks):
        accs = [decade_topk[str(d)].get(str(k), 0) for d in decades]
        ax.plot(labels, accs, marker="o", color=colors[i % len(colors)],
                label=f"Top-{k}", linewidth=2, markersize=5)

    ax.set_xlabel("Decade")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Top-K Country Accuracy by Decade ({split} split)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{split}_topk_decade_accuracy.png", dpi=150)
    plt.close(fig)


def _plot_per_country(results: Dict[str, Any], plots_dir: Path, split: str):
    per_country_acc = results.get("per_country_accuracy", {})
    per_country_counts = results.get("per_country_counts", {})
    if not per_country_acc:
        return

    countries = sorted(per_country_acc, key=lambda c: per_country_acc[c])
    accs = np.array([per_country_acc[c] for c in countries])
    counts = [per_country_counts.get(c, 0) for c in countries]

    fig, ax = plt.subplots(figsize=(8, max(6, len(countries) * 0.28)))
    colors = plt.cm.RdYlGn(accs / 100.0)
    bars = ax.barh(countries, accs, color=colors, edgecolor="none")
    ax.set_xlabel("Country Accuracy (%)")
    ax.set_title(f"Per-Country Accuracy ({split} split)")
    ax.set_xlim(0, 120)
    ax.axvline(results["country_accuracy"], color="steelblue", linestyle="--",
               linewidth=1.2, label=f"Overall ({results['country_accuracy']:.1f}%)")
    ax.legend(loc="lower right")

    for bar, val, n in zip(bars, accs, counts):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}% (n={n})", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(plots_dir / f"{split}_per_country_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_world_map(results: Dict[str, Any], plots_dir: Path, split: str):
    if not _HAS_GEOPANDAS:
        print("geopandas not installed — skipping world map (pip install geopandas)")
        return

    try:
        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        except AttributeError:
            import geodatasets
            world = gpd.read_file(geodatasets.get_path("naturalearth.lowres"))
    except Exception as e:
        print(f"Could not load world map data ({e}) — skipping world map")
        return

    per_country_acc = results.get("per_country_accuracy", {})
    if not per_country_acc:
        return

    name_lower = {n.lower(): n for n in per_country_acc}
    aliases = {
        "united states of america": "United States",
        "russian federation": "Russia",
        "republic of korea": "South Korea",
        "democratic people's republic of korea": "North Korea",
        "iran (islamic republic of)": "Iran",
        "syrian arab republic": "Syria",
        "viet nam": "Vietnam",
        "czechia": "Czech Republic",
        "czech republic": "Czechia",
        "côte d'ivoire": "Ivory Coast",
        "congo, dem. rep.": "Democratic Republic of the Congo",
    }

    def match(geo_name):
        low = geo_name.lower()
        if low in name_lower:
            return per_country_acc[name_lower[low]]
        resolved = aliases.get(low, "").lower()
        if resolved in name_lower:
            return per_country_acc[name_lower[resolved]]
        return None

    world["accuracy"] = world["name"].apply(match)

    fig, ax = plt.subplots(figsize=(16, 8))
    world[world["accuracy"].isna()].plot(ax=ax, color="#e0e0e0", edgecolor="white", linewidth=0.3)
    world[world["accuracy"].notna()].plot(
        ax=ax, column="accuracy", cmap="RdYlGn", vmin=0, vmax=100,
        edgecolor="white", linewidth=0.3, legend=True,
        legend_kwds={"label": "Country Accuracy (%)", "shrink": 0.6},
    )
    ax.set_title(f"Country Accuracy by Region ({split} split)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{split}_world_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(results: Dict[str, Any], plots_dir: Path, split: str, top_n: int = 20):
    confusion = results.get("confusion", {})
    per_country_counts = results.get("per_country_counts", {})
    if not confusion:
        return

    top_countries = sorted(per_country_counts, key=lambda c: -per_country_counts[c])[:top_n]
    n = len(top_countries)
    idx = {c: i for i, c in enumerate(top_countries)}

    matrix = np.zeros((n, n))
    for true_c in top_countries:
        row = confusion.get(true_c, {})
        total = per_country_counts.get(true_c, 0)
        if total == 0:
            continue
        for pred_c, count in row.items():
            if pred_c in idx:
                matrix[idx[true_c], idx[pred_c]] = count / total * 100

    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), max(8, n * 0.55)))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="% of true class predicted as column")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(top_countries, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(top_countries, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — Top {n} Countries by Count ({split} split)")

    for i in range(n):
        val = matrix[i, i]
        color = "white" if val > 55 else "black"
        ax.text(i, i, f"{val:.0f}%", ha="center", va="center", fontsize=7,
                color=color, fontweight="bold")

    fig.tight_layout()
    fig.savefig(plots_dir / f"{split}_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- public API ---------------------------------------------------------------

def plot_results(results: Dict[str, Any], plots_dir: Path, split: str, confusion_top_n: int = 20):
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_decade_accuracy(results, plots_dir, split)
    _plot_overall_accuracy(results, plots_dir, split)
    _plot_topk_decade(results, plots_dir, split)
    _plot_per_country(results, plots_dir, split)
    _plot_world_map(results, plots_dir, split)
    _plot_confusion_matrix(results, plots_dir, split, top_n=confusion_top_n)


def plot_model_comparison(
    results1: Dict[str, Any], name1: str,
    results2: Dict[str, Any], name2: str,
    plots_dir: Path,
):
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_decades = sorted(
        {int(d) for d in results1["decade_accuracy"]} |
        {int(d) for d in results2["decade_accuracy"]}
    )
    labels = [f"{d}s" for d in all_decades]

    def accs_for(results):
        return [results["decade_accuracy"].get(str(d), None) for d in all_decades]

    accs1 = accs_for(results1)
    accs2 = accs_for(results2)

    # Decade accuracy comparison
    fig, ax = plt.subplots(figsize=(11, 5))

    def _draw(accs, name, color):
        plot_vals = [a if a is not None else np.nan for a in accs]
        ax.plot(labels, plot_vals, marker="o", color=color, label=name, linewidth=2, markersize=5)
        valid_xs = [i for i, a in enumerate(accs) if a is not None]
        valid_ys = [a for a in accs if a is not None]
        if len(valid_xs) >= 2:
            trend, m = _trend_line(valid_xs, valid_ys)
            trend_vals = [np.nan] * len(labels)
            for xi, tv in zip(valid_xs, trend):
                trend_vals[xi] = tv
            direction = "+" if m > 0 else ""
            ax.plot(labels, trend_vals, color=color, linestyle="--", linewidth=1, alpha=0.6,
                    label=f"{name} trend ({direction}{m:.1f}%/decade)")

    _draw(accs1, name1, "steelblue")
    _draw(accs2, name2, "darkorange")

    ax.set_xlabel("Decade")
    ax.set_ylabel("Country Accuracy (%)")
    ax.set_title("Model Comparison: Decade Accuracy")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "comparison_decade_accuracy.png", dpi=150)
    plt.close(fig)

    # Overall accuracy side-by-side
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    w = 0.35
    vals1 = [results1["country_accuracy"], results1["us_state_accuracy"]]
    vals2 = [results2["country_accuracy"], results2["us_state_accuracy"]]
    bars1 = ax.bar(x - w / 2, vals1, w, label=name1, color="steelblue", edgecolor="white")
    bars2 = ax.bar(x + w / 2, vals2, w, label=name2, color="darkorange", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["Country", "US State"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Comparison: Overall Accuracy")
    ax.set_ylim(0, 100)
    ax.legend()
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(plots_dir / "comparison_overall_accuracy.png", dpi=150)
    plt.close(fig)


# --- CLI ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset split.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt model checkpoint")
    parser.add_argument("--model_name", default=None, help="Name tag — outputs go into a subdirectory with this name")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--metrics_path", default=None, help="JSON path to save metrics (confusion excluded from CLI print)")
    parser.add_argument("--plots_dir", default=None, help="Directory to save plots")
    parser.add_argument("--compare_checkpoint", default=None, help="Second checkpoint for LP vs FT comparison plots")
    parser.add_argument("--compare_model_name", default=None, help="Name for the second model")
    parser.add_argument("--confusion_top_n", type=int, default=20, help="Countries shown in confusion matrix")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ensure_dataset(config)
    mappings = build_label_mappings(ds["train"])
    num_countries = len(mappings["country"]["mapping"])
    num_us_states = len(mappings["us_state"]["mapping"])

    model = load_model(Path(args.checkpoint), num_countries, num_us_states, config, device)
    dataloader = build_dataloader(config, args.split, args.batch_size)
    results = evaluate(model, dataloader, mappings)

    # Print summary (exclude the verbose confusion dict)
    printable = {k: v for k, v in results.items() if k != "confusion"}
    print(json.dumps(printable, indent=2))

    plots_dir = Path(args.plots_dir) / args.model_name if args.plots_dir and args.model_name else \
                Path(args.plots_dir) if args.plots_dir else None

    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        if args.model_name:
            metrics_path = metrics_path.parent / args.model_name / metrics_path.name
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as fp:
            json.dump(results, fp, indent=2)

    if plots_dir:
        plot_results(results, plots_dir, args.split, confusion_top_n=args.confusion_top_n)
        print(f"Plots saved to {plots_dir}")

    if args.compare_checkpoint:
        model2 = load_model(Path(args.compare_checkpoint), num_countries, num_us_states, config, device)
        results2 = evaluate(model2, dataloader, mappings)
        name1 = args.model_name or Path(args.checkpoint).stem
        name2 = args.compare_model_name or Path(args.compare_checkpoint).stem
        compare_dir = Path(args.plots_dir) if args.plots_dir else Path("plots")
        plot_model_comparison(results, name1, results2, name2, compare_dir)
        print(f"Comparison plots saved to {compare_dir}")


if __name__ == "__main__":
    main()
