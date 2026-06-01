import torch
import torch.nn as nn
import torchvision.models as models

class BobTheBuilder(nn.Module):
  def __init__(self, num_countries, num_us_states, config):
    super().__init__()
    
    self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    hidden_dim = self.vit.heads.head.in_features
    self.vit.heads = nn.Identity()

    # freeze backbone if needed
    if config.get("model", {}).get("experiment_type", "finetune") == "linear_probe":
      for p in self.vit.parameters():
        p.requires_grad = False
    # else: parameters remain trainable for full fine-tuning

    # classification heads for country, continent, and US region
    def make_head(out_dim, hidden=512, dropout=0.2):
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

  def freeze_backbone(self):
    for p in self.vit.parameters():
      p.requires_grad = False

  def unfreeze_backbone(self):
    for p in self.vit.parameters():
      p.requires_grad = True

  def forward(self, x, return_features=False):
    features = self.vit(x)

    country_logits = self.country_head(features)
    us_state_logits = self.us_state_head(features)

    outputs = {
      "country": country_logits,
      "us_state": us_state_logits,
    }

    if return_features:
      return outputs, features
    
    return outputs



