import { Building, ModelPrediction } from "./types";
import { COUNTRIES } from "./countries";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function reverseGeocode(lat: number, lng: number): Promise<string | null> {
  try {
    const res = await fetch(`${API_BASE}/api/reverse-geocode?lat=${lat}&lng=${lng}`);
    if (!res.ok) return null;
    const data: { country: string | null } = await res.json();
    return data.country ?? null;
  } catch {
    return null;
  }
}

export async function fetchRandomBuilding(): Promise<Building> {
  const res = await fetch(`${API_BASE}/api/buildings/random`);
  if (!res.ok) throw new Error(`Failed to fetch building: ${res.status}`);
  return res.json();
}

export async function fetchPrediction(building: Building): Promise<ModelPrediction> {
  try {
    const res = await fetch(
      `${API_BASE}/api/predict/${building.id}?model=finetune`,
      { method: "POST" }
    );
    if (!res.ok) throw new Error(`Prediction failed: ${res.status}`);
    const data: ModelPrediction = await res.json();

    // Backend always returns 0,0 — map predicted country code to centroid
    const match = COUNTRIES.find(
      (c) =>
        c.code === data.countryCode ||
        c.name.toLowerCase() === data.country.toLowerCase()
    );
    if (match) data.coordinates = { lat: match.lat, lng: match.lng };

    return data;
  } catch {
    return _mockPrediction(building);
  }
}

function _mockPrediction(building: Building): ModelPrediction {
  const others = COUNTRIES.filter((c) => c.code !== building.countryCode);
  const pick = others[Math.floor(Math.random() * Math.min(30, others.length))];
  return {
    country: pick.name,
    countryCode: pick.code,
    continent: pick.continent,
    coordinates: { lat: pick.lat, lng: pick.lng },
    confidence: { continent: 0.68, country: 0.45 },
    features: [
      { label: "Architectural style", value: "International Modern", confidence: 0.81, delayMs: 800 },
      { label: "Facade material", value: "Concrete & Glass", confidence: 0.73, delayMs: 1600 },
      { label: "Climate indicators", value: "Temperate", confidence: 0.65, delayMs: 2400 },
      { label: "Regional pattern", value: `${pick.continent} typology`, confidence: 0.71, delayMs: 3200 },
    ],
    gradcamUrl: null,
    usState: null,
  };
}
