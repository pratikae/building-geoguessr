# Building GeoGuessr

This repo trains a model to predict geography from building imagery.

## Setup

1. Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Download and extract the dataset into the repo root:

```bash
cd /homes/iws/pve2/building-geoguessr
curl -L -o builtidentity_dataset.zip https://homes.cs.washington.edu/~albertdu/builtidentity_dataset.zip
unzip -o builtidentity_dataset.zip
```

The dataset should extract into `resplit_year_guessr_dataset`.

## Run the pipeline

Train the model with:

```bash
python3 train.py
```

The code uses `config.yaml` for training settings. By default, the dataset loader expects `resplit_year_guessr_dataset` in the repo root.

## Notes

- The pipeline uses a Vision Transformer backbone from `torchvision`.
- If you want to run a full fine-tune instead of a linear probe, update `config.yaml`:

```yaml
model:
  experiment_type: "finetune"
```

- Model checkpoints are saved as `best_linear_probe_model.pt` or `best_finetune_model.pt`.

## Evaluation

After training, run the evaluation script on the `test` split (post-1945 buildings):

```bash
python3 evaluate.py --checkpoint best_finetune_model.pt --split test --metrics_path metrics/test_metrics.json
```

The script reports:
- `country_accuracy`
- `us_state_accuracy`
- `decade_accuracy`

If training a linear probe instead, use `best_linear_probe_model.pt` as the checkpoint.

## GradCAM comparison

Use `gradcam.py` to compare where the linear-probe and fine-tuned models attend on the same inputs:

```bash
python3 gradcam.py --lp_weights best_linear_probe_model.pt --ft_weights best_finetune_model.pt --split validation --output_dir gradcam_results --n_top 20
```

The `--n_top` option controls how many of the most divergent images are visualized. This generates top-ranked divergence visualizations that help show how the models learn differently.
