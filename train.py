import argparse
import importlib
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import BobTheBuilder


def parse_args():
  parser = argparse.ArgumentParser(description="Train BobTheBuilder")
  parser.add_argument("--config", type=str, default="config.yaml", help="path to config file")
  parser.add_argument("--lr", type=float, help="override learning rate for hyperparameter tuning")
  parser.add_argument("--batch_size", type=int, help="override batch size for tuning")
  parser.add_argument("--epochs", type=int, help="override number of training epochs")
  parser.add_argument("--experiment_type", type=str, choices=["linear_probe", "finetune"], help="override experiment type")
  parser.add_argument("--weight_decay", type=float, help="override weight decay for AdamW")
  parser.add_argument("--metrics_path", type=str, default=None, help="optional JSON file to save final metrics")
  return parser.parse_args()


def load_config(config_path):
  with open(config_path, "r") as f:
    return yaml.safe_load(f)


def get_dataloaders(config):
  try:
    dataset_module = importlib.import_module("dataset")
  except ModuleNotFoundError as exc:
    raise RuntimeError(
      "dataset.py not found. Please add a dataset module that implements get_dataloaders(config)."
    ) from exc

  if not hasattr(dataset_module, "get_dataloaders"):
    raise RuntimeError("dataset.py must implement get_dataloaders(config)")

  return dataset_module.get_dataloaders(config)


def get_label_cardinalities(config):
  model_cfg = config.get("model", {})
  if model_cfg.get("num_countries") is not None:
    return (
      model_cfg["num_countries"],
      model_cfg["num_us_states"],
    )

  try:
    dataset_module = importlib.import_module("dataset")
    if hasattr(dataset_module, "get_label_cardinalities"):
      return dataset_module.get_label_cardinalities(config)
  except ModuleNotFoundError:
    pass

  raise RuntimeError(
    "Label cardinalities must be provided in config.yaml under model or by dataset.get_label_cardinalities()"
  )


def compute_loss(predictions, targets, config, device):
  criterion = nn.CrossEntropyLoss()

  loss_country = criterion(predictions["country"], targets["country"].to(device))

  if "is_us" in targets:
    is_us_mask = (targets["is_us"] == 1).to(device)
  else:
    is_us_mask = torch.zeros_like(targets["country"]).to(device)

  if is_us_mask.sum() > 0:
    us_logits = predictions["us_state"][is_us_mask]
    us_targets = targets["us_state"][is_us_mask].to(device)
    loss_us = criterion(us_logits, us_targets)
  else:
    loss_us = torch.tensor(0.0, device=device)

  w = config["training"]["loss_weights"]
  total_loss = (
    w["country"] * loss_country
    + w["us_state"] * loss_us
  )
  return total_loss, {
    "country": loss_country.item(),
    "us_state": loss_us.item(),
  }


def train_epoch(model, dataloader, optimizer, config, device):
    model.train()
    running_loss = 0.0

    if len(dataloader) == 0:
        return 0.0

    pbar = tqdm(dataloader, desc="Training")
    for images, targets in pbar:
        images = images.to(device)

        optimizer.zero_grad()
        predictions = model(images)

        loss, _ = compute_loss(predictions, targets, config, device)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": running_loss / (pbar.n + 1)})

    return running_loss / len(dataloader)


def validate(model, dataloader, config, device):
    model.eval()
    running_loss = 0.0
    country_correct = 0
    total_samples = 0

    if len(dataloader) == 0:
        return 0.0, 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            predictions = model(images)

            loss, _ = compute_loss(predictions, targets, config, device)
            running_loss += loss.item()

            preds = torch.argmax(predictions["country"], dim=1)
            country_correct += (preds == targets["country"].to(device)).sum().item()
            total_samples += targets["country"].size(0)

    val_loss = running_loss / len(dataloader)
    val_acc = (country_correct / total_samples) * 100 if total_samples > 0 else 0.0
    return val_loss, val_acc


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.experiment_type is not None:
        config["model"]["experiment_type"] = args.experiment_type
    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(config)
    num_countries, num_us_states = get_label_cardinalities(config)

    model = BobTheBuilder(num_countries, num_us_states, config)
    model.to(device)

    if config["model"]["experiment_type"] == "linear_probe":
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )

    best_val_acc = 0.0
    epochs = config["training"]["epochs"]

    print(f"starting training: {config['model']['experiment_type'].upper()}")
    for epoch in range(1, epochs + 1):
        print(f"\nepoch {epoch}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, config, device)
        val_loss, val_acc = validate(model, val_loader, config, device)

        print(f"train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_{config['model']['experiment_type']}_model.pt")
            print("new best model saved!")

    if args.metrics_path:
        import json

        Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_path, "w") as f:
            json.dump(
                {
                    "best_val_acc": best_val_acc,
                    "final_val_loss": val_loss,
                    "final_val_acc": val_acc,
                    "epochs": epochs,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
