"""
BOB Backend — FastAPI with ViT-B/16 inference
"""
import io
import json
import random
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model import BobTheBuilder  # noqa: E402

# Startup

with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_COUNTRIES = 147
NUM_US_STATES = 53

with open(ROOT / "models" / "label_mappings.json") as f:
    _raw = json.load(f)

COUNTRY_IDX2NAME: dict[int, str] = {int(k): v for k, v in _raw["country"].items()}
US_STATE_IDX2NAME: dict[int, str] = {int(k): v for k, v in _raw["us_state"].items()}

try:
    import pycountry as _pycountry
    _PYCOUNTRY = _pycountry
except ImportError:
    _PYCOUNTRY = None

# alpha-2 -> continent
_CONTINENT: dict[str, str] = {
    "AF": "Africa", "AL": "Europe", "DZ": "Africa", "AO": "Africa", "AQ": "Antarctica",
    "AR": "South America", "AM": "Asia", "AU": "Oceania", "AT": "Europe", "AZ": "Asia",
    "BD": "Asia", "BY": "Europe", "BE": "Europe", "BJ": "Africa", "BT": "Asia",
    "BO": "South America", "BA": "Europe", "BW": "Africa", "BR": "South America",
    "BN": "Asia", "BG": "Europe", "BF": "Africa", "BI": "Africa", "KH": "Asia",
    "CM": "Africa", "CA": "North America", "CF": "Africa", "TD": "Africa",
    "CL": "South America", "CN": "Asia", "CO": "South America", "KM": "Africa",
    "CG": "Africa", "CD": "Africa", "CR": "North America", "HR": "Europe",
    "CU": "North America", "CY": "Asia", "CZ": "Europe", "DK": "Europe",
    "DJ": "Africa", "DO": "North America", "EC": "South America", "EG": "Africa",
    "SV": "North America", "GQ": "Africa", "ER": "Africa", "EE": "Europe",
    "SZ": "Africa", "ET": "Africa", "FJ": "Oceania", "FI": "Europe", "FR": "Europe",
    "GA": "Africa", "GM": "Africa", "GE": "Asia", "DE": "Europe", "GH": "Africa",
    "GR": "Europe", "GT": "North America", "GN": "Africa", "GW": "Africa",
    "GY": "South America", "HT": "North America", "HN": "North America",
    "HU": "Europe", "IS": "Europe", "IN": "Asia", "ID": "Asia", "IR": "Asia",
    "IQ": "Asia", "IE": "Europe", "IL": "Asia", "IT": "Europe", "JM": "North America",
    "JP": "Asia", "JO": "Asia", "KZ": "Asia", "KE": "Africa", "KI": "Oceania",
    "KP": "Asia", "KR": "Asia", "KW": "Asia", "KG": "Asia", "LA": "Asia",
    "LV": "Europe", "LB": "Asia", "LS": "Africa", "LR": "Africa", "LY": "Africa",
    "LT": "Europe", "LU": "Europe", "MG": "Africa", "MW": "Africa", "MY": "Asia",
    "MV": "Asia", "ML": "Africa", "MT": "Europe", "MR": "Africa", "MU": "Africa",
    "MX": "North America", "MD": "Europe", "MN": "Asia", "ME": "Europe",
    "MA": "Africa", "MZ": "Africa", "MM": "Asia", "NA": "Africa", "NP": "Asia",
    "NL": "Europe", "NZ": "Oceania", "NI": "North America", "NE": "Africa",
    "NG": "Africa", "MK": "Europe", "NO": "Europe", "OM": "Asia", "PK": "Asia",
    "PA": "North America", "PG": "Oceania", "PY": "South America", "PE": "South America",
    "PH": "Asia", "PL": "Europe", "PT": "Europe", "QA": "Asia", "RO": "Europe",
    "RU": "Europe", "RW": "Africa", "SA": "Asia", "SN": "Africa", "RS": "Europe",
    "SC": "Africa", "SL": "Africa", "SG": "Asia", "SK": "Europe", "SI": "Europe",
    "SB": "Oceania", "SO": "Africa", "ZA": "Africa", "SS": "Africa", "ES": "Europe",
    "LK": "Asia", "SD": "Africa", "SR": "South America", "SE": "Europe",
    "CH": "Europe", "SY": "Asia", "TW": "Asia", "TJ": "Asia", "TZ": "Africa",
    "TH": "Asia", "TL": "Asia", "TG": "Africa", "TO": "Oceania",
    "TT": "North America", "TN": "Africa", "TR": "Asia", "TM": "Asia",
    "UG": "Africa", "UA": "Europe", "AE": "Asia", "GB": "Europe",
    "US": "North America", "UY": "South America", "UZ": "Asia", "VU": "Oceania",
    "VE": "South America", "VN": "Asia", "YE": "Asia", "ZM": "Africa", "ZW": "Africa",
}


def _country_meta(name: str) -> tuple[str, str]:
    """Return (iso_alpha2, continent) for a country name string."""
    if _PYCOUNTRY is not None:
        try:
            c = _PYCOUNTRY.countries.lookup(name)
            return c.alpha_2, _CONTINENT.get(c.alpha_2, "Unknown")
        except LookupError:
            pass
    return "XX", "Unknown"


TRANSFORMS = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_model(filename: str) -> BobTheBuilder:
    state = torch.load(ROOT / "models" / filename, map_location=DEVICE, weights_only=True)
    num_countries = state["country_head.4.weight"].shape[0]
    num_us_states = state["us_state_head.4.weight"].shape[0]
    m = BobTheBuilder(num_countries, num_us_states, CONFIG)
    m.load_state_dict(state)
    m.to(DEVICE)
    m.eval()
    return m


print(f"Loading models on {DEVICE}...")
MODELS: dict[str, BobTheBuilder] = {
    "finetune": _load_model("best_finetune_model.pt"),
    "linear_probe": _load_model("best_linear_probe_model.pt"),
}
print("Models ready.")


def _run_inference(image_bytes: bytes, model_key: str = "finetune") -> dict:
    model = MODELS.get(model_key)
    if model is None:
        raise ValueError(f"Unknown model '{model_key}'. Choose 'finetune' or 'linear_probe'.")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = TRANSFORMS(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)

    country_probs = F.softmax(outputs["country"], dim=1)[0]
    us_probs = F.softmax(outputs["us_state"], dim=1)[0]

    top_country_idx = int(country_probs.argmax())
    top_country_conf = float(country_probs[top_country_idx])
    top_country_name = COUNTRY_IDX2NAME.get(top_country_idx) or "Unknown"

    top_us_idx = int(us_probs.argmax())
    top_us_conf = float(us_probs[top_us_idx])
    top_us_name = US_STATE_IDX2NAME.get(top_us_idx) or "Unknown"

    # top-5 country predictions
    top5_vals, top5_idxs = country_probs.topk(5)
    top5 = [
        {"country": COUNTRY_IDX2NAME.get(int(i)) or "Unknown", "confidence": float(v)}
        for i, v in zip(top5_idxs, top5_vals)
    ]

    return {
        "country": top_country_name,
        "country_confidence": top_country_conf,
        "us_state": top_us_name,
        "us_state_confidence": top_us_conf,
        "top5_countries": top5,
    }


# Schemas

class Coordinates(BaseModel):
    lat: float
    lng: float


class DetectedFeature(BaseModel):
    label: str
    value: str
    confidence: float
    delayMs: int


class ModelConfidence(BaseModel):
    continent: float
    country: float


class PredictionResponse(BaseModel):
    country: str
    countryCode: str
    continent: str
    coordinates: Coordinates
    confidence: ModelConfidence
    features: list[DetectedFeature]
    gradcamUrl: Optional[str] = None
    usState: Optional[str] = None


class Building(BaseModel):
    id: str
    imageUrl: str
    name: str
    country: str
    countryCode: str
    coordinates: Coordinates
    yearBuilt: int
    continent: str


def _make_prediction_response(result: dict) -> PredictionResponse:
    country_name = result["country"]
    code, continent = _country_meta(country_name)
    country_conf = result["country_confidence"]

    is_us = code == "US"
    features: list[DetectedFeature] = [
        DetectedFeature(label="Top prediction", value=country_name, confidence=round(country_conf, 3), delayMs=800),
    ]
    for rank, entry in enumerate(result["top5_countries"][1:], start=2):
        features.append(DetectedFeature(
            label=f"#{rank} candidate",
            value=entry["country"],
            confidence=round(entry["confidence"], 3),
            delayMs=800 + rank * 400,
        ))
    if is_us:
        features.append(DetectedFeature(
            label="US State",
            value=result["us_state"] or "Unknown",
            confidence=round(result["us_state_confidence"], 3),
            delayMs=3200,
        ))

    return PredictionResponse(
        country=country_name,
        countryCode=code,
        continent=continent,
        coordinates=Coordinates(lat=0.0, lng=0.0),
        confidence=ModelConfidence(continent=round(country_conf, 3), country=round(country_conf, 3)),
        features=features,
        gradcamUrl=None,
        usState=result["us_state"] if is_us else None,
    )


# App

app = FastAPI(title="BOB API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"

MOCK_BUILDINGS = [
    Building(id="1", imageUrl="https://upload.wikimedia.org/wikipedia/commons/a/a0/Sydney_Australia._(21652882908).jpg", name="Sydney Opera House", country="Australia", countryCode="AU", coordinates=Coordinates(lat=-33.8568, lng=151.2153), yearBuilt=1973, continent="Oceania"),
    Building(id="2", imageUrl="https://upload.wikimedia.org/wikipedia/commons/6/6f/Palace_of_the_Parliament_%28HDR%29.jpg", name="Palace of the Parliament", country="Romania", countryCode="RO", coordinates=Coordinates(lat=44.4275, lng=26.0878), yearBuilt=1997, continent="Europe"),
    Building(id="3", imageUrl="https://upload.wikimedia.org/wikipedia/commons/c/c3/Burj_Khalifa.jpg", name="Burj Khalifa", country="United Arab Emirates", countryCode="AE", coordinates=Coordinates(lat=25.1972, lng=55.2744), yearBuilt=2010, continent="Asia"),
    Building(id="4", imageUrl="https://upload.wikimedia.org/wikipedia/commons/4/47/New_york_times_square-terabass.jpg", name="One Times Square", country="United States", countryCode="US", coordinates=Coordinates(lat=40.758, lng=-73.9855), yearBuilt=1904, continent="North America"),
    Building(id="5", imageUrl="https://upload.wikimedia.org/wikipedia/commons/9/9a/Big_Ben_2013.jpg", name="Elizabeth Tower", country="United Kingdom", countryCode="GB", coordinates=Coordinates(lat=51.5007, lng=-0.1246), yearBuilt=1859, continent="Europe"),
]

_BUILDINGS_BY_ID = {b.id: b for b in MOCK_BUILDINGS}


def _fetch_url(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


@app.get("/")
def root():
    return {"status": "ok", "service": "BOB API", "device": str(DEVICE)}


@app.get("/api/buildings/random", response_model=Building)
def get_random_building():
    return random.choice(MOCK_BUILDINGS)


@app.get("/api/buildings/{building_id}", response_model=Building)
def get_building(building_id: str):
    b = _BUILDINGS_BY_ID.get(building_id)
    if not b:
        raise HTTPException(status_code=404, detail="Building not found")
    return b


@app.post("/api/predict/{building_id}", response_model=PredictionResponse)
def predict_location(
    building_id: str,
    model: str = Query(default="finetune", description="'finetune' or 'linear_probe'"),
):
    b = _BUILDINGS_BY_ID.get(building_id)
    if not b:
        raise HTTPException(status_code=404, detail="Building not found")
    try:
        image_bytes = _fetch_url(b.imageUrl)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch image: {exc}")
    try:
        result = _run_inference(image_bytes, model_key=model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _make_prediction_response(result)

@app.post("/api/predict-image", response_model=PredictionResponse)
async def predict_from_upload(
    file: UploadFile = File(...),
    model: str = Query(default="finetune", description="'finetune' or 'linear_probe'"),
):
    """Upload an image and run inference."""
    contents = await file.read()
    try:
        result = _run_inference(contents, model_key=model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Inference failed: {exc}")

    return _make_prediction_response(result)


@app.post("/api/predict-url", response_model=PredictionResponse)
def predict_from_url(
    image_url: str = Query(..., description="Publicly accessible image URL"),
    model: str = Query(default="finetune", description="'finetune' or 'linear_probe'"),
):
    """Fetch an image from a URL and run inference."""
    try:
        image_bytes = _fetch_url(image_url)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to fetch image: {exc}")

    try:
        result = _run_inference(image_bytes, model_key=model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return _make_prediction_response(result)
