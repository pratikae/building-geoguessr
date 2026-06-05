# Building GeoGuessr

**Beat Bob** — a GeoGuessr-style game where you compete against a ViT-B/16 model trained on pre-1945 architecture. Drop a pin on the globe and see if you can outguess Bob the Builder.

Research question: *Does architecture still have a regional accent after globalization?*

---

## Repo structure

```
building-geoguessr/
├── model.py          # ViT-B/16 model with hierarchical classification head
├── train.py          # Training (linear probe + fine-tune)
├── tune.py           # Hyperparameter search
├── evaluate.py       # Evaluation on post-1945 test split
├── gradcam.py        # Grad-CAM comparison between LP and FT models
├── data.py           # Data utilities
├── dataset.py        # Dataset class (HuggingFace + re-split by year)
├── config.yaml       # Training config
├── requirements.txt  # ML Python deps
├── backend/          # FastAPI server — bridges model to game
│   ├── main.py
│   └── requirements.txt
└── frontend/         # Next.js game (Beat Bob)
```

---

## ML pipeline

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset

```bash
curl -L -o builtidentity_dataset.zip https://homes.cs.washington.edu/~albertdu/builtidentity_dataset.zip
unzip -o builtidentity_dataset.zip
```

Dataset extracts into `resplit_year_guessr_dataset`. Pre-1945 buildings are train/val; post-1945 are test.

### 3. Train

```bash
python3 train.py
```

Config is in `config.yaml`. For full fine-tune instead of linear probe:

```yaml
model:
  experiment_type: "finetune"
```

Checkpoints: `best_linear_probe_model.pt` / `best_finetune_model.pt`

### 4. Tune

```bash
python3 tune.py --search grid --n_trials 20 --metrics_dir tuning_metrics --output tuning_results.csv
```

### 5. Evaluate

```bash
python3 evaluate.py \
  --checkpoint best_finetune_model.pt \
  --model_name finetune_v1 \
  --split test \
  --metrics_path metrics/metrics.json \
  --plots_dir plots/
```

To compare two models (e.g. linear probe vs fine-tune):

```bash
python3 evaluate.py \
  --checkpoint best_finetune_model.pt \
  --model_name finetune \
  --compare_checkpoint best_linear_probe_model.pt \
  --compare_model_name linear_probe \
  --split test \
  --plots_dir plots/
```

**Flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | — | Path to `.pt` model checkpoint (required) |
| `--model_name` | — | Name tag — per-model outputs go into `plots/<name>/` and `metrics/<name>/` |
| `--split` | `test` | `train`, `validation`, or `test` |
| `--batch_size` | `64` | Batch size |
| `--metrics_path` | — | JSON path to save full metrics (including confusion data) |
| `--plots_dir` | — | Directory to save plots |
| `--compare_checkpoint` | — | Second `.pt` checkpoint for side-by-side comparison plots |
| `--compare_model_name` | — | Name for the second model |
| `--confusion_top_n` | `20` | Number of countries shown in the confusion matrix |

**Plots produced per model** (saved under `plots/<model_name>/`):

| File | Description |
|------|-------------|
| `{split}_decade_accuracy.png` | Country accuracy per decade with linear trend line and sample counts |
| `{split}_overall_accuracy.png` | Country vs US state accuracy summary |
| `{split}_topk_decade_accuracy.png` | Top-1 / Top-3 / Top-5 accuracy by decade — shows whether region is still detectable even when exact country is wrong |
| `{split}_per_country_accuracy.png` | Horizontal bar chart of every country sorted by accuracy, coloured green→red |
| `{split}_world_map.png` | Choropleth map of per-country accuracy (requires `geopandas`) |
| `{split}_confusion_matrix.png` | Row-normalised confusion matrix for the top N countries by count |

**Comparison plots** (saved directly in `plots/` when `--compare_checkpoint` is given):

| File | Description |
|------|-------------|
| `comparison_decade_accuracy.png` | Both models' decade accuracy + trend lines on one chart |
| `comparison_overall_accuracy.png` | Side-by-side country / US state accuracy bars |

> **World map note:** requires `geopandas` (`pip install geopandas`). If not installed the plot is skipped and the rest run normally.

### 6. Grad-CAM

```bash
python3 gradcam.py \
  --lp_weights best_linear_probe_model.pt \
  --ft_weights best_finetune_model.pt \
  --split validation \
  --output_dir gradcam_results \
  --n_top 20
```

---

## Backend (FastAPI)

Serves the model to the game frontend.

### Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Runs on `http://localhost:8000`. Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/buildings/random` | Random post-1945 building |
| `GET` | `/api/buildings/{id}` | Specific building |
| `POST` | `/api/predict/{building_id}` | Model prediction + Grad-CAM |

### Plugging in the real model

All model inference goes in `backend/main.py → predict_location()`. Look for `# TODO` comments. Steps:

1. Load the building image (from dataset URL or cache)
2. Preprocess: resize to 224×224, normalize with ImageNet stats
3. Run forward pass through the trained model
4. Softmax → predicted class + confidence
5. Reverse-geocode class to lat/lng centroid
6. Run Grad-CAM → save PNG → set `gradcamUrl` in response

```python
# In predict_location():
from your_model import load_model, predict, run_gradcam

model = load_model("best_finetune_model.pt")
prediction = predict(model, image_url)
gradcam_path = run_gradcam(model, image_url)
```

---

## Frontend (Beat Bob game)

### Setup

```bash
cd frontend
npm install
npm run dev
```

Opens at `http://localhost:3000`. The game auto-connects to the backend at `localhost:8000`.

To point at a different backend URL:

```bash
NEXT_PUBLIC_API_URL=https://your-backend.com npm run dev
```

### Game flow

1. **Landing** — scrapbook collage intro, flip card with Bob
2. **Game** — building photo (left) + interactive globe (right), 5 rounds
3. **Guess** — click globe to place pin, optionally pick country from dropdown
4. **Analysis** — "Bob is thinking" overlay shows predicted features + confidence
5. **Results** — arcs on globe, score comparison (you vs Bob), cumulative total

### Key files

```
frontend/
├── app/page.tsx              # Landing page (scrapbook + flip card)
├── app/game/page.tsx         # Game page (full layout)
├── components/globe/         # react-globe.gl wrapper + custom pin
├── components/game/          # ModelAnalysis, RoundResults, CountrySelector
├── lib/api.ts                # Fetch wrappers (+ mock fallback if backend down)
├── lib/types.ts              # Shared TypeScript types
└── public/                   # Bob images, pin, scrapbook PNGs
```
