"""
ARCHLOC Backend — FastAPI

Stub endpoints ready for the real ViT-B/16 model to be plugged in.
Your partner should replace the mock logic in `predict_location` with
the actual model inference + Grad-CAM generation.
"""

import random
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ARCHLOC API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

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


class Building(BaseModel):
    id: str
    imageUrl: str
    name: str
    country: str
    countryCode: str
    coordinates: Coordinates
    yearBuilt: int
    continent: str


# ---------------------------------------------------------------------------
# Mock dataset — replace with HuggingFace dataset loader
# ---------------------------------------------------------------------------

MOCK_BUILDINGS = [
    Building(
        id="1",
        imageUrl="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Sydney_Australia._(21652882908).jpg/1280px-Sydney_Australia._(21652882908).jpg",
        name="Sydney Opera House",
        country="Australia",
        countryCode="AU",
        coordinates=Coordinates(lat=-33.8568, lng=151.2153),
        yearBuilt=1973,
        continent="Oceania",
    ),
    Building(
        id="2",
        imageUrl="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Palace_of_the_Parliament_%28HDR%29.jpg/1280px-Palace_of_the_Parliament_%28HDR%29.jpg",
        name="Palace of the Parliament",
        country="Romania",
        countryCode="RO",
        coordinates=Coordinates(lat=44.4275, lng=26.0878),
        yearBuilt=1997,
        continent="Europe",
    ),
    Building(
        id="3",
        imageUrl="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Burj_Khalifa.jpg/440px-Burj_Khalifa.jpg",
        name="Burj Khalifa",
        country="United Arab Emirates",
        countryCode="AE",
        coordinates=Coordinates(lat=25.1972, lng=55.2744),
        yearBuilt=2010,
        continent="Asia",
    ),
    Building(
        id="4",
        imageUrl="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/New_york_times_square-terabass.jpg/1280px-New_york_times_square-terabass.jpg",
        name="One Times Square",
        country="United States",
        countryCode="US",
        coordinates=Coordinates(lat=40.758, lng=-73.9855),
        yearBuilt=1904,
        continent="North America",
    ),
    Building(
        id="5",
        imageUrl="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9a/Big_Ben_2013.jpg/800px-Big_Ben_2013.jpg",
        name="Elizabeth Tower",
        country="United Kingdom",
        countryCode="GB",
        coordinates=Coordinates(lat=51.5007, lng=-0.1246),
        yearBuilt=1859,
        continent="Europe",
    ),
]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "ARCHLOC API"}


@app.get("/api/buildings/random", response_model=Building)
def get_random_building():
    """Return a random post-1945 building from the dataset."""
    return random.choice(MOCK_BUILDINGS)


@app.get("/api/buildings/{building_id}", response_model=Building)
def get_building(building_id: str):
    """Return a specific building by ID."""
    for b in MOCK_BUILDINGS:
        if b.id == building_id:
            return b
    raise HTTPException(status_code=404, detail="Building not found")


@app.post("/api/predict/{building_id}", response_model=PredictionResponse)
def predict_location(building_id: str):
    """
    Run the ViT-B/16 model on the building image and return a prediction.

    TODO: Replace mock response with:
      1. Load the building image from dataset / URL
      2. Preprocess: resize, normalize, convert to tensor
      3. Run model forward pass → logits for continent, country, US region
      4. Apply softmax → get predicted class + confidence
      5. Reverse-geocode predicted class to lat/lng centroid
      6. Run Grad-CAM on the image → save heatmap PNG, return URL
    """
    building = next((b for b in MOCK_BUILDINGS if b.id == building_id), None)
    if not building:
        raise HTTPException(status_code=404, detail="Building not found")

    # --- MOCK PREDICTION ---
    # Pretend to get something slightly wrong
    wrong_map = {
        "AU": ("New Zealand", "NZ", "Oceania", -40.9006, 174.886),
        "RO": ("Bulgaria", "BG", "Europe", 42.7339, 25.4858),
        "AE": ("Saudi Arabia", "SA", "Asia", 23.8859, 45.0792),
        "US": ("Canada", "CA", "North America", 56.1304, -106.3468),
        "GB": ("Ireland", "IE", "Europe", 53.1424, -7.6921),
    }
    guess = wrong_map.get(building.countryCode)
    if guess:
        country, code, continent, lat, lng = guess
    else:
        country = building.country
        code = building.countryCode
        continent = building.continent
        lat = building.coordinates.lat + random.uniform(-5, 5)
        lng = building.coordinates.lng + random.uniform(-5, 5)

    return PredictionResponse(
        country=country,
        countryCode=code,
        continent=continent,
        coordinates=Coordinates(lat=lat, lng=lng),
        confidence=ModelConfidence(
            continent=round(random.uniform(0.70, 0.95), 3),
            country=round(random.uniform(0.40, 0.85), 3),
        ),
        features=[
            DetectedFeature(label="Architectural style", value="Modernist / International", confidence=0.88, delayMs=800),
            DetectedFeature(label="Facade material", value="Concrete & Glass", confidence=0.79, delayMs=1400),
            DetectedFeature(label="Climate indicators", value="Temperate coastal", confidence=0.72, delayMs=2000),
            DetectedFeature(label="Urban density", value="High — central metro", confidence=0.65, delayMs=2600),
            DetectedFeature(label="Regional pattern", value=f"{continent} typology", confidence=0.91, delayMs=3200),
        ],
        gradcamUrl=None,  # Set to URL of generated Grad-CAM PNG when model is real
    )


@app.post("/api/predict-image", response_model=PredictionResponse)
async def predict_from_upload(file: UploadFile = File(...)):
    """
    Accept an uploaded image and return a prediction.
    Used when serving images directly rather than from dataset URLs.

    TODO: Same implementation notes as predict_location above.
    """
    contents = await file.read()
    # Pass `contents` bytes to model inference pipeline
    raise HTTPException(status_code=501, detail="Upload inference not yet implemented — plug in the model here")
