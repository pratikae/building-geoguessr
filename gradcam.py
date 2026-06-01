import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import (
    HuggingFaceImageDataset,
    build_label_mappings,
    default_transforms,
    ensure_dataset,
    resolve_image_column,
)
from model import BobTheBuilder


VIT_GRID = 14  # 224px / 16px patch size


def reshape_transform(tensor):
    result = tensor[:, 1:, :].reshape(tensor.size(0), VIT_GRID, VIT_GRID, tensor.size(2))
    return result.permute(0, 3, 1, 2)


class CountryWrapper(torch.nn.Module):
    """Thin wrapper so GradCAM receives a plain tensor output instead of a dict."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["country"]


def load_model(weights_path, num_countries, num_us_states, config, device):
    model = BobTheBuilder(num_countries, num_us_states, config)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model


def score_all(lp_wrapper, ft_wrapper, dataset, device, batch_size):
    """Returns (scores, true_classes) arrays indexed by dataset position.

    Score = L1 distance between the two models' country softmax distributions.
    Higher score = models disagree more on this image.
    """
    scores, true_classes = [], []
    lp_wrapper.eval()
    ft_wrapper.eval()

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            indices = range(start, min(start + batch_size, len(dataset)))
            images  = torch.stack([dataset[i][0] for i in indices]).to(device)
            labels  = [dataset[i][1]["country"] for i in indices]

            lp_probs = F.softmax(lp_wrapper(images), dim=-1)
            ft_probs = F.softmax(ft_wrapper(images), dim=-1)
            l1 = (lp_probs - ft_probs).abs().sum(dim=-1)

            scores.extend(l1.cpu().tolist())
            true_classes.extend(labels)

            if (start // batch_size + 1) % 10 == 0:
                print(f"  scored {min(start + batch_size, len(dataset))}/{len(dataset)}")

    return np.array(scores), np.array(true_classes)


def make_cam(wrapper, target_layers, image_tensor, target_class, device):
    cam = GradCAM(model=wrapper, target_layers=target_layers, reshape_transform=reshape_transform)
    grayscale = cam(
        input_tensor=image_tensor.unsqueeze(0).to(device),
        targets=[ClassifierOutputTarget(target_class)],
    )
    return grayscale[0]


def to_rgb(image_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = image_tensor.cpu().numpy().transpose(1, 2, 0) * std + mean
    return np.clip(img, 0, 1).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description="GradCAM: top-N most-divergent images between linear probe and finetune")
    p.add_argument("--config",      default="config.yaml")
    p.add_argument("--lp_weights",  default="best_linear_probe_model.pt")
    p.add_argument("--ft_weights",  default="best_finetune_model.pt")
    p.add_argument("--n_top",       type=int, default=20, help="number of images to visualize")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--output_dir",  default="gradcam_results")
    p.add_argument("--split",       default="validation", choices=["train", "validation", "test"])
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds        = ensure_dataset(config)
    split_ds  = ds[args.split]
    image_col = resolve_image_column(split_ds)
    mappings  = build_label_mappings(split_ds)

    idx_to_country = {v: k for k, v in mappings["country"]["mapping"].items()}
    num_countries  = len(mappings["country"]["mapping"])
    num_us_states  = len(mappings["us_state"]["mapping"])

    image_size = config.get("data", {}).get("image_size", 224)
    dataset    = HuggingFaceImageDataset(split_ds, mappings, image_col, default_transforms(image_size))

    lp_model = load_model(args.lp_weights, num_countries, num_us_states, config, device)
    ft_model = load_model(args.ft_weights, num_countries, num_us_states, config, device)
    lp_wrapper = CountryWrapper(lp_model)
    ft_wrapper = CountryWrapper(ft_model)

    print(f"Scoring {len(dataset)} images on {args.split} split...")
    scores, true_classes = score_all(lp_wrapper, ft_wrapper, dataset, device, args.batch_size)

    top_indices = np.argsort(scores)[::-1][:args.n_top]

    target_layers_lp = [lp_model.vit.encoder.layers[-1].ln_1]
    target_layers_ft = [ft_model.vit.encoder.layers[-1].ln_1]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating GradCAM for top {args.n_top} images...")
    for rank, idx in enumerate(top_indices, 1):
        idx = int(idx)
        image_tensor, targets = dataset[idx]
        true_class = targets["country"]
        rgb = to_rgb(image_tensor)

        lp_cam     = make_cam(lp_wrapper, target_layers_lp, image_tensor, true_class, device)
        ft_cam     = make_cam(ft_wrapper, target_layers_ft, image_tensor, true_class, device)
        lp_overlay = show_cam_on_image(rgb, lp_cam, use_rgb=True)
        ft_overlay = show_cam_on_image(rgb, ft_cam, use_rgb=True)

        with torch.no_grad():
            lp_probs = F.softmax(lp_wrapper(image_tensor.unsqueeze(0).to(device)), dim=-1)[0]
            ft_probs = F.softmax(ft_wrapper(image_tensor.unsqueeze(0).to(device)), dim=-1)[0]

        lp_pred = lp_probs.argmax().item()
        ft_pred = ft_probs.argmax().item()

        true_label = idx_to_country.get(true_class, str(true_class))
        lp_label   = f"{idx_to_country.get(lp_pred, str(lp_pred))} ({lp_probs[lp_pred]:.1%})"
        ft_label   = f"{idx_to_country.get(ft_pred, str(ft_pred))} ({ft_probs[ft_pred]:.1%})"

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, img, title in zip(axes,
                                  [rgb, lp_overlay, ft_overlay],
                                  [f"Original\nTrue: {true_label}",
                                   f"Linear Probe\n{lp_label}",
                                   f"Finetune\n{ft_label}"]):
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        fig.suptitle(f"Rank {rank}  |  L1 divergence: {scores[idx]:.3f}", fontsize=12)
        plt.tight_layout()
        out_path = output_dir / f"rank_{rank:03d}_idx_{idx}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{rank}/{args.n_top}] true={true_label} | lp={lp_label} | ft={ft_label}")

    print(f"\nDone. Saved {args.n_top} images to {output_dir}/")


if __name__ == "__main__":
    main()
