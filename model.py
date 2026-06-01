import torch
import torch.nn as nn
import torchvision.models as models
from typing import Any, Dict, Tuple, Union


class BobTheBuilder(nn.Module):
  def __init__(self, num_countries: int, num_us_states: int, config: Dict[str, Any]):
    super().__init__()

    # instantiate ViT and handle different torchvision versions better
    try:
      self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    except Exception:
      # fallback for older torchvision where pretrained arg is used
      self.vit = models.vit_b_16(pretrained=True)

    # determine feature dimension produced by the backbone's head
    hidden_dim = None
    # common torchvision layouts: vit.heads.head.in_features, vit.head.in_features, vit.classifier.in_features
    if hasattr(self.vit, "heads"):
      heads = getattr(self.vit, "heads")
      if hasattr(heads, "head") and hasattr(heads.head, "in_features"):
        hidden_dim = heads.head.in_features
      elif hasattr(heads, "in_features"):
        hidden_dim = heads.in_features

    if hidden_dim is None and hasattr(self.vit, "head") and hasattr(self.vit.head, "in_features"):
      hidden_dim = self.vit.head.in_features

    if hidden_dim is None and hasattr(self.vit, "classifier") and hasattr(self.vit.classifier, "in_features"):
      hidden_dim = self.vit.classifier.in_features

    if hidden_dim is None:
      raise RuntimeError(
        "Unable to determine ViT head feature size; update `model.py` for your torchvision version."
      )

    # replace the classification head with identity so backbone returns features
    if hasattr(self.vit, "heads"):
      setattr(self.vit, "heads", nn.Identity())
    elif hasattr(self.vit, "head"):
      setattr(self.vit, "head", nn.Identity())
    elif hasattr(self.vit, "classifier"):
      setattr(self.vit, "classifier", nn.Identity())

    # freeze backbone if needed
    if config.get("model", {}).get("experiment_type", "finetune") == "linear_probe":
      for p in self.vit.parameters():
        p.requires_grad = False

    # classification heads for country and US state
    def make_head(out_dim: int, hidden: int = 512, dropout: float = 0.2) -> nn.Module:
      if hidden is None or hidden <= 0:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))
      return nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
      )

    self.country_head = make_head(num_countries, hidden=512)
    self.us_state_head = make_head(num_us_states, hidden=128)

  def freeze_backbone(self) -> None:
    for p in self.vit.parameters():
      p.requires_grad = False

  def unfreeze_backbone(self) -> None:
    for p in self.vit.parameters():
      p.requires_grad = True

  def forward(
    self, x: torch.Tensor, return_features: bool = False
  ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
    features = self.vit(x)

    country_logits = self.country_head(features)
    us_state_logits = self.us_state_head(features)

    outputs: Dict[str, torch.Tensor] = {
      "country": country_logits,
      "us_state": us_state_logits,
    }

    if return_features:
      return outputs, features

    return outputs



