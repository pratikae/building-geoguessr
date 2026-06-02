from pathlib import Path
from typing import Dict, Optional

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from datasets import load_from_disk

DEFAULT_DATASET_PATH = "resplit_year_guessr_dataset"
IMAGE_COLUMN_CANDIDATES = [
    "image",
    "img",
    "photo",
    "Image",
    "streetview",
    "streetview_image",
    "image_path",
    "img_path",
]
LABEL_COLUMN_CANDIDATES = {
    "country": ["Country", "country"],
    "us_state": ["State", "state"],
}


def resolve_column(dataset, candidates):
    for name in candidates:
        if name in dataset.column_names:
            return name
    return None


def resolve_image_column(dataset):
    for candidate in IMAGE_COLUMN_CANDIDATES:
        if candidate in dataset.column_names:
            return candidate

    for name, feature in dataset.features.items():
        if feature.__class__.__name__ == "Image":
            return name

    raise RuntimeError(
        "No image column found in the dataset. Add an image feature column or update "
        "IMAGE_COLUMN_CANDIDATES in dataset.py to match your dataset schema."
    )


def ensure_dataset(config: Dict) -> object:
    data_cfg = config.get("data", {})
    dataset_path = Path(data_cfg.get("path", DEFAULT_DATASET_PATH))

    if dataset_path.exists():
        return load_from_disk(str(dataset_path))

    raise RuntimeError(
        f"Dataset not found at '{dataset_path}'. Run `python data.py` first to build it."
    )


REQUIRED_LABELS = {"country", "us_state"}

def build_label_mappings(dataset):
    label_cols = {}
    for label, candidates in LABEL_COLUMN_CANDIDATES.items():
        column_name = resolve_column(dataset, candidates)
        if column_name is None:
            if label in REQUIRED_LABELS:
                raise RuntimeError(
                    f"Cannot find column for label '{label}'. Expected one of {candidates}. "
                    "Please verify your dataset or add the missing column."
                )
            continue
        label_cols[label] = column_name

    mappings = {}
    for label, column_name in label_cols.items():
        values = dataset.unique(column_name)
        if hasattr(dataset.features[column_name], "names"):
            mapping = {name: idx for idx, name in enumerate(dataset.features[column_name].names)}
        else:
            unique_values = set(values)
            if None in unique_values:
                unique_values.remove(None)
                mapping = {value: idx for idx, value in enumerate(sorted(unique_values))}
                mapping[None] = len(mapping)
            else:
                mapping = {value: idx for idx, value in enumerate(sorted(unique_values))}
        mappings[label] = {
            "column": column_name,
            "mapping": mapping,
        }

    return mappings


def get_label_cardinalities(config: Dict):
    ds = ensure_dataset(config)
    mappings = build_label_mappings(ds["train"])
    return (
        len(mappings["country"]["mapping"]),
        len(mappings["us_state"]["mapping"]),
    )


def compute_class_weights(dataset, label_column: str, num_classes: int):
    """Compute inverse frequency weights to handle class imbalance.
    
    Args:
        dataset: HuggingFace dataset
        label_column: Column name for the labels
        num_classes: Total number of classes
    
    Returns:
        List of weights, normalized so that sum = num_classes
    """
    from collections import Counter
    
    # Get label distribution
    labels = dataset[label_column]
    label_counts = Counter(labels)
    
    # Compute weights as inverse frequency
    weights = []
    total_samples = len(dataset)
    for class_idx in range(num_classes):
        count = label_counts.get(class_idx, 1)  # Avoid division by zero
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    return weights


def get_class_weights_for_training(config: Dict) -> Dict[str, list]:
    """Compute class weights from training dataset to handle imbalance.
    
    Returns dict with 'country_class_weights' and 'us_state_class_weights'.
    """
    ds = ensure_dataset(config)
    mappings = build_label_mappings(ds["train"])
    train_dataset = ds["train"]
    
    result = {}
    
    # Country weights
    country_col = mappings["country"]["column"]
    country_weights = compute_class_weights(
        train_dataset, 
        country_col, 
        len(mappings["country"]["mapping"])
    )
    result["country_class_weights"] = country_weights
    
    # US state weights
    us_state_col = mappings["us_state"]["column"]
    us_state_weights = compute_class_weights(
        train_dataset,
        us_state_col,
        len(mappings["us_state"]["mapping"])
    )
    result["us_state_class_weights"] = us_state_weights
    
    return result


def default_transforms(image_size: int = 224):
    return T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def augmented_transforms(image_size: int = 224):
    """Stronger augmentation for training with limited data."""
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class HuggingFaceImageDataset(Dataset):
    def __init__(self, hf_dataset, label_config, image_column: str, transforms=None):
        self.dataset = hf_dataset
        self.label_config = label_config
        self.image_column = image_column
        self.transforms = transforms or default_transforms()

    def __len__(self):
        return len(self.dataset)

    def _load_image(self, image_value):
        if isinstance(image_value, str):
            return Image.open(image_value).convert("RGB")

        if hasattr(image_value, "to_pil"):
            return image_value.to_pil()

        if isinstance(image_value, dict) and "path" in image_value:
            return Image.open(image_value["path"]).convert("RGB")

        if isinstance(image_value, Image.Image):
            return image_value.convert("RGB")

        raise RuntimeError(
            f"Unsupported image value type: {type(image_value)} for column '{self.image_column}'."
        )

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self._load_image(sample[self.image_column])
        image = self.transforms(image)

        targets = {}
        for label, config in self.label_config.items():
            value = sample[config["column"]]
            mapping = config["mapping"]
            if isinstance(value, int):
                targets[label] = value
            else:
                if value not in mapping:
                    if None in mapping:
                        targets[label] = mapping[None]
                    else:
                        raise RuntimeError(
                            f"Unexpected label value '{value}' for column '{config['column']}'."
                        )
                else:
                    targets[label] = mapping[value]

        if "is_us" in sample:
            targets["is_us"] = int(sample["is_us"])

        return image, targets


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images)

    collated = {}
    for key in targets[0].keys():
        collated[key] = torch.tensor([t[key] for t in targets], dtype=torch.long)
    return images, collated


def get_dataloaders(config: Dict):
    ds = ensure_dataset(config)
    image_column = resolve_image_column(ds["train"])
    mappings = build_label_mappings(ds["train"])

    data_cfg = config.get("data", {})
    image_size = data_cfg.get("image_size", 224)
    batch_size = config["training"]["batch_size"]

    # Use augmented transforms for training, regular transforms for validation
    train_transforms = augmented_transforms(image_size)
    val_transforms = default_transforms(image_size)

    train_ds = HuggingFaceImageDataset(ds["train"], mappings, image_column, train_transforms)
    val_ds = HuggingFaceImageDataset(ds["validation"], mappings, image_column, val_transforms)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn),
    )
