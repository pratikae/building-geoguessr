import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

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
    }
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset split.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint file")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--metrics_path", default=None, help="Optional JSON path to save metrics")
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
        Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_path, "w") as fp:
            json.dump(results, fp, indent=2)


if __name__ == "__main__":
    main()
