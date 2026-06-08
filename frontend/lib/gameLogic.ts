import { Coordinates } from "./types";

const EARTH_RADIUS_KM = 6371;

export function haversineDistance(a: Coordinates, b: Coordinates): number {
  const toRad = (x: number) => (x * Math.PI) / 180;
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const s =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.sin(dLng / 2) ** 2;
  return 2 * EARTH_RADIUS_KM * Math.asin(Math.sqrt(s));
}

export function calculateScore(distanceKm: number): number {
  if (distanceKm <= 0) return 5000;
  return Math.round(5000 * Math.exp(-distanceKm / 2000));
}

export function formatScore(score: number): string {
  return score.toLocaleString();
}

export function formatDistance(km: number): string {
  if (km < 1) return `${Math.round(km * 1000)} m`;
  if (km < 100) return `${km.toFixed(1)} km`;
  return `${Math.round(km).toLocaleString()} km`;
}

export function totalScore(rounds: { score: number }[]): number {
  return rounds.reduce((sum, r) => sum + r.score, 0);
}
