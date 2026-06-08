export interface Coordinates {
  lat: number;
  lng: number;
}

export interface Building {
  id: string;
  imageUrl: string;
  name: string;
  country: string;
  countryCode: string;
  coordinates: Coordinates;
  yearBuilt: number;
  continent: string;
}

export interface DetectedFeature {
  label: string;
  value: string;
  confidence: number;
  delayMs: number;
}

export interface ModelConfidence {
  continent: number;
  country: number;
}

export interface ModelPrediction {
  country: string;
  countryCode: string;
  continent: string;
  coordinates: Coordinates;
  confidence: ModelConfidence;
  features: DetectedFeature[];
  gradcamUrl: string | null;
  usState: string | null;
}

export type GamePhase = "guessing" | "analyzing" | "results" | "gameover";

export interface Country {
  code: string;
  name: string;
  lat: number;
  lng: number;
  continent: string;
}

export interface GuessResult {
  country: string;
  coordinates: Coordinates;
  distanceKm: number;
  score: number;
}

export interface RoundScore {
  round: number;
  buildingName: string;
  trueCountry: string;
  trueCoordinates: Coordinates;
  userGuess: GuessResult;
  modelGuess: GuessResult;
}

export interface GlobePin {
  lat: number;
  lng: number;
  color: string;
  label: string;
  size: number;
}

export interface GlobeArc {
  startLat: number;
  startLng: number;
  endLat: number;
  endLng: number;
  color: string[];
}
