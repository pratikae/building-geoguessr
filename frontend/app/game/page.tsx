"use client";

import { useState, useCallback, useEffect } from "react";
import { Send, RotateCcw, ChevronLeft, Trophy, Cpu, MapPin } from "lucide-react";
import Link from "next/link";
import Image from "next/image";

import { Building, Coordinates, Country, GamePhase, RoundScore, GlobePin } from "@/lib/types";
import { COUNTRIES } from "@/lib/countries";
import { haversineDistance, calculateScore, formatScore, totalScore } from "@/lib/gameLogic";
import { fetchRandomBuilding, fetchPrediction, reverseGeocode } from "@/lib/api";

import GlobeWrapper from "@/components/globe/GlobeWrapper";
import CountrySelector from "@/components/game/CountrySelector";
import RoundResults from "@/components/game/RoundResults";

const TOTAL_ROUNDS = 5;
const MONO = "var(--font-geist-mono)";

// â”€â”€ Final screen â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function FinalScreen({ scores, onRestart }: { scores: RoundScore[]; onRestart: () => void }) {
  const userTotal = totalScore(scores.map((s) => ({ score: s.userGuess.score })));
  const modelTotal = totalScore(scores.map((s) => ({ score: s.modelGuess.score })));
  const userWins = scores.filter((s) => s.userGuess.score >= s.modelGuess.score).length;
  const userWon = userTotal >= modelTotal;

  return (
    <div className="h-full w-full flex flex-col items-center justify-center gap-10 px-6" style={{ background: "#080806" }}>
      <div style={{ color: "#64748b", fontFamily: MONO, fontSize: "0.9rem", letterSpacing: "0.2em" }}>GAME OVER</div>
      <h1 className="text-center font-bold" style={{ fontSize: "clamp(3rem, 7vw, 5.5rem)", color: userWon ? "#10b981" : "#7c3aed", lineHeight: 1.1 }}>
        {userWon ? "You beat Bob." : "Bob wins this time."}
      </h1>
      <p style={{ color: "#64748b", fontFamily: MONO, fontSize: "1.05rem" }}>{userWins} of {TOTAL_ROUNDS} rounds won</p>

      <div className="w-full rounded-2xl" style={{ maxWidth: 520, background: "#111109", border: "1px solid rgba(212,160,23,0.12)", padding: "2.4rem" }}>
        <div className="flex justify-between items-start mb-10">
          <div>
            <div style={{ fontSize: "2.6rem", fontWeight: 700, color: "#3b82f6", fontFamily: MONO, lineHeight: 1 }}>{formatScore(userTotal)}</div>
            <div style={{ color: "#64748b", fontFamily: MONO, fontSize: "0.85rem", letterSpacing: "0.15em", marginTop: 8 }}>YOUR SCORE</div>
          </div>
          <div style={{ color: "#2d3748", fontSize: "1.5rem", marginTop: 8 }}>vs</div>
          <div className="text-right">
            <div style={{ fontSize: "2.6rem", fontWeight: 700, color: "#7c3aed", fontFamily: MONO, lineHeight: 1 }}>{formatScore(modelTotal)}</div>
            <div style={{ color: "#64748b", fontFamily: MONO, fontSize: "0.85rem", letterSpacing: "0.15em", marginTop: 8 }}>BOB&apos;S SCORE</div>
          </div>
        </div>
        <div className="flex flex-col gap-2.5">
          {scores.map((s, i) => {
            const uWon = s.userGuess.score >= s.modelGuess.score;
            return (
              <div key={i} className="flex items-center justify-between rounded-xl px-5 py-3.5" style={{ background: "#1a1a12" }}>
                <span style={{ color: "#64748b", fontFamily: MONO, fontSize: "0.92rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 200 }}>R{i + 1}: {s.buildingName}</span>
                <div className="flex gap-4 items-center">
                  <span style={{ color: uWon ? "#3b82f6" : "#64748b", fontFamily: MONO, fontSize: "1rem" }}>{formatScore(s.userGuess.score)}</span>
                  <span style={{ color: "#2d3748" }}>vs</span>
                  <span style={{ color: !uWon ? "#7c3aed" : "#64748b", fontFamily: MONO, fontSize: "1rem" }}>{formatScore(s.modelGuess.score)}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="flex gap-4">
        <button onClick={onRestart} className="flex items-center gap-3 rounded-xl font-bold transition-all" style={{ padding: "18px 40px", fontSize: "1.05rem", background: "linear-gradient(135deg, rgba(212,160,23,0.15), rgba(124,58,237,0.15))", border: "1px solid rgba(212,160,23,0.45)", color: "#d4a017", fontFamily: MONO }}>
          <RotateCcw size={18} /> Play Again
        </button>
        <Link href="/" className="flex items-center rounded-xl font-bold" style={{ padding: "18px 40px", fontSize: "1.05rem", border: "1px solid rgba(255,255,255,0.07)", color: "#64748b", fontFamily: MONO }}>
          Home
        </Link>
      </div>
    </div>
  );
}

// â”€â”€ Header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function TopHeader({ round, totalRounds, userScore, modelScore }: { round: number; totalRounds: number; userScore: number; modelScore: number }) {
  return (
    <div className="flex-shrink-0 flex items-center justify-between z-30" style={{ padding: "1rem 2rem", background: "rgba(5,5,9,0.92)", borderBottom: "1px solid rgba(212,160,23,0.08)", backdropFilter: "blur(12px)" }}>
      <div className="flex items-center gap-5">
        <Link href="/" className="flex items-center gap-2 transition-colors" style={{ color: "#64748b", fontFamily: MONO, fontSize: "0.95rem" }}>
          <ChevronLeft size={18} /> Exit
        </Link>
        <div style={{ width: 1, height: 22, background: "rgba(212,160,23,0.12)" }} />
        <div className="flex items-center gap-2.5">
          <Image src="/bob_logo.png" alt="Bob" width={44} height={44} unoptimized className="rounded-lg" style={{ objectFit: "cover" }} />
          <span className="font-bold" style={{ fontFamily: MONO, fontSize: "1.15rem", color: "#e2e8f0", letterSpacing: "0.06em" }}>Beat Bob</span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {Array.from({ length: totalRounds }, (_, i) => (
          <div key={i} className="rounded-full transition-all duration-300" style={{ width: i === round - 1 ? 36 : 22, height: 9, background: i < round - 1 ? "#d4a017" : i === round - 1 ? "rgba(212,160,23,0.7)" : "rgba(212,160,23,0.12)", boxShadow: i === round - 1 ? "0 0 10px rgba(212,160,23,0.6)" : "none" }} />
        ))}
        <span style={{ marginLeft: 6, color: "#64748b", fontFamily: MONO, fontSize: "0.95rem" }}>Round {round} of {totalRounds}</span>
      </div>

      <div className="flex items-center gap-8">
        <div className="flex items-center gap-2.5">
          <Trophy size={18} style={{ color: "#3b82f6" }} />
          <span style={{ color: "#64748b", fontFamily: MONO, fontSize: "0.95rem" }}>You</span>
          <span className="font-bold tabular-nums" style={{ color: "#e2e8f0", fontFamily: MONO, fontSize: "1.2rem" }}>{userScore.toLocaleString()}</span>
        </div>
        <div style={{ width: 1, height: 22, background: "rgba(255,255,255,0.06)" }} />
        <div className="flex items-center gap-2.5">
          <Cpu size={18} style={{ color: "#7c3aed" }} />
          <span style={{ color: "#64748b", fontFamily: MONO, fontSize: "0.95rem" }}>Bob</span>
          <span className="font-bold tabular-nums" style={{ color: "#e2e8f0", fontFamily: MONO, fontSize: "1.2rem" }}>{modelScore.toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
}

// â”€â”€ Photo panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

function PhotoPanel({ building }: { building: Building }) {
  const [lightboxOpen, setLightboxOpen] = useState(false);

  return (
    <>
      <div className="relative w-full h-full overflow-hidden group" style={{ background: "#080806" }}>
        {/* Corner accents */}
        {["top-4 left-4 border-t-2 border-l-2", "top-4 right-4 border-t-2 border-r-2", "bottom-4 left-4 border-b-2 border-l-2", "bottom-4 right-4 border-b-2 border-r-2"].map((cls, k) => (
          <div key={k} className={`absolute w-7 h-7 z-20 pointer-events-none ${cls}`} style={{ borderColor: "rgba(212,160,23,0.5)" }} />
        ))}

        <div className="scanlines absolute inset-0 z-10 pointer-events-none" />

        <Image
          src={building.imageUrl}
          alt="Mystery building"
          fill
          className="object-cover transition-transform duration-700 group-hover:scale-[1.03]"
          sizes="40vw"
          priority
          unoptimized
        />

        {/* Big visible bottom prompt */}
        <div
          className="absolute bottom-0 left-0 right-0 z-20"
          style={{ background: "linear-gradient(transparent, rgba(5,5,9,0.98) 40%)", padding: "3rem 1.8rem 1.8rem" }}
        >
          <div
            className="font-bold mb-2"
            style={{ color: "#ffffff", fontSize: "1.6rem", lineHeight: 1.2, letterSpacing: "0.01em" }}
          >
            Where in the world
            <br />is this building?
          </div>
          <div style={{ color: "#64748b", fontFamily: MONO, fontSize: "0.95rem" }}>
            Built {building.yearBuilt} · Drop a pin on the globe
          </div>
        </div>

        {/* Year badge */}
        <div className="absolute top-5 left-5 z-20" style={{ padding: "10px 18px", borderRadius: 10, background: "rgba(5,5,9,0.9)", border: "1px solid rgba(212,160,23,0.3)", color: "#d4a017", fontFamily: MONO, fontSize: "1rem", letterSpacing: "0.08em" }}>
          {building.yearBuilt}
        </div>

        {/* Expand button */}
        <button
          onClick={() => setLightboxOpen(true)}
          className="absolute top-5 right-5 z-20 opacity-0 group-hover:opacity-100 transition-opacity"
          style={{ padding: "10px 16px", borderRadius: 10, background: "rgba(5,5,9,0.9)", border: "1px solid rgba(212,160,23,0.2)", color: "#d4a017", fontFamily: MONO, fontSize: "0.9rem" }}
        >
          Expand
        </button>
      </div>

      {lightboxOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-8" style={{ background: "rgba(5,5,9,0.97)" }} onClick={() => setLightboxOpen(false)}>
          <div className="relative w-full max-w-5xl aspect-video rounded-2xl overflow-hidden">
            <Image src={building.imageUrl} alt="Mystery building" fill className="object-contain" unoptimized />
          </div>
        </div>
      )}
    </>
  );
}

// â”€â”€ Main game page â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

export default function GamePage() {
  const [phase, setPhase] = useState<GamePhase>("guessing");
  const [round, setRound] = useState(1);
  const [building, setBuilding] = useState<Building | null>(null);
  const [userPin, setUserPin] = useState<Coordinates | null>(null);
  const [userCountry, setUserCountry] = useState<string | null>(null);
  const [scores, setScores] = useState<RoundScore[]>([]);
  const [globeFocus, setGlobeFocus] = useState<{ lat: number; lng: number; altitude: number } | undefined>();


  const userTotal = totalScore(scores.map((s) => ({ score: s.userGuess.score })));
  const modelTotal = totalScore(scores.map((s) => ({ score: s.modelGuess.score })));

  useEffect(() => {
    setBuilding(null);
    setUserPin(null);
    setUserCountry(null);
    setGlobeFocus(undefined);
    fetchRandomBuilding().then(setBuilding);
  }, [round]);

  // Click: place / move pin
  const handleGlobeClick = useCallback((lat: number, lng: number) => {
    if (phase !== "guessing") return;
    setUserPin({ lat, lng });
  }, [phase]);

  const handleCountrySelect = useCallback((country: Country) => {
    setUserCountry(country.code);
    setUserPin({ lat: country.lat, lng: country.lng });
    setGlobeFocus({ lat: country.lat, lng: country.lng, altitude: 1.8 });
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!userPin || !building) return;
    const [prediction, geocoded] = await Promise.all([
      fetchPrediction(building),
      reverseGeocode(userPin.lat, userPin.lng),
    ]);
    const sel = COUNTRIES.find((c) => c.code === userCountry);
    const guessCountryName = geocoded ?? sel?.name ?? "Unknown";
    const userDist = haversineDistance(userPin, building.coordinates);
    const modelDist = haversineDistance(prediction.coordinates, building.coordinates);
    setScores((prev) => [...prev, {
      round, buildingName: building.name, trueCountry: building.country,
      userGuess: { country: guessCountryName, coordinates: userPin, distanceKm: userDist, score: calculateScore(userDist) },
      modelGuess: { country: prediction.country, coordinates: prediction.coordinates, distanceKm: modelDist, score: calculateScore(modelDist) },
    }]);
    setPhase("results");
  }, [userPin, building, userCountry, round]);

  const handleNext = useCallback(() => {
    if (round >= TOTAL_ROUNDS) setPhase("gameover");
    else { setRound((r) => r + 1); setPhase("guessing"); }
  }, [round]);

  const handleRestart = useCallback(() => {
    setRound(1); setPhase("guessing"); setScores([]);
  }, []);

  if (phase === "gameover") return <FinalScreen scores={scores} onRestart={handleRestart} />;

  const currentRoundScore = scores[scores.length - 1] ?? null;
  const canSubmit = !!userPin && phase === "guessing";

  // Custom pin image is used during guessing; results screen uses coloured dots via pins=
  const globePins: GlobePin[] = [];

  return (
    <div className="h-full flex flex-col" style={{ background: "#080806" }}>
      <TopHeader round={round} totalRounds={TOTAL_ROUNDS} userScore={userTotal} modelScore={modelTotal} />

      {/* Split: photo 40% | globe 60% */}
      <div className="flex-1 flex min-h-0">

        {/* Photo */}
        <div className="flex-shrink-0" style={{ width: "40%", borderRight: "1px solid rgba(212,160,23,0.07)" }}>
          {building && <PhotoPanel building={building} />}
        </div>

        {/* Globe */}
        <div className="flex-1 relative" style={{ overflow: "hidden" }}>

          {/* Hint (no pin placed yet) */}
          {phase === "guessing" && !userPin && (
            <div className="absolute top-7 left-1/2 z-10 pointer-events-none" style={{ transform: "translateX(-50%)", whiteSpace: "nowrap" }}>
              <div className="flex items-center gap-3" style={{ padding: "14px 30px", borderRadius: 999, background: "rgba(5,5,9,0.92)", border: "1px solid rgba(212,160,23,0.22)", backdropFilter: "blur(10px)" }}>
                <MapPin size={20} style={{ color: "#d4a017" }} />
                <span style={{ color: "#a0aec0", fontFamily: MONO, fontSize: "1.05rem" }}>
                  Click the globe to place your pin
                </span>
              </div>
            </div>
          )}

          {/* Pin-placed sub-hint */}
          {phase === "guessing" && userPin && (
            <div className="absolute top-7 left-1/2 z-10 pointer-events-none" style={{ transform: "translateX(-50%)", whiteSpace: "nowrap" }}>
              <div className="flex items-center gap-2.5" style={{ padding: "10px 22px", borderRadius: 999, background: "rgba(5,5,9,0.85)", border: "1px solid rgba(212,160,23,0.15)", backdropFilter: "blur(8px)" }}>
                <div className="w-2.5 h-2.5 rounded-full" style={{ background: "#3b82f6", boxShadow: "0 0 8px #3b82f6" }} />
                <span style={{ color: "#64748b", fontFamily: MONO, fontSize: "0.92rem" }}>
                  Click to reposition your pin · scroll to zoom
                </span>
              </div>
            </div>
          )}

          {/* Globe */}
          <div className="absolute inset-0">
            <GlobeWrapper
              pins={globePins}
              customPin={userPin}
              onGlobeClick={handleGlobeClick}
              interactive
              autoRotate={false}
              focusLat={globeFocus?.lat}
              focusLng={globeFocus?.lng}
              altitude={globeFocus?.altitude ?? 2.5}
            />
          </div>

          {/* Guess controls â€” floating bottom-right */}
          {phase === "guessing" && (
            <div className="absolute bottom-8 right-8 z-10 flex flex-col gap-3" style={{ width: 340 }}>
              {/* Coords */}
              {userPin && (
                <div className="text-center" style={{ padding: "13px 20px", borderRadius: 14, background: "rgba(5,5,9,0.94)", border: "1px solid rgba(212,160,23,0.25)", backdropFilter: "blur(10px)", color: "#d4a017", fontFamily: MONO, fontSize: "1.05rem", letterSpacing: "0.06em" }}>
                  {Math.abs(userPin.lat).toFixed(2)}&deg;{userPin.lat >= 0 ? "N" : "S"}&nbsp;&nbsp;
                  {Math.abs(userPin.lng).toFixed(2)}&deg;{userPin.lng >= 0 ? "E" : "W"}
                </div>
              )}

              {/* Country selector card */}
              <div style={{ background: "rgba(5,5,9,0.94)", border: "1px solid rgba(212,160,23,0.18)", borderRadius: 20, padding: "22px 20px", backdropFilter: "blur(10px)" }}>
                <div style={{ color: "#64748b", fontFamily: MONO, fontSize: "0.88rem", letterSpacing: "0.18em", marginBottom: 14 }}>
                  WHERE IS THIS BUILDING?
                </div>
                <CountrySelector value={userCountry} onChange={handleCountrySelect} />
              </div>

              {/* Submit */}
              <button
                onClick={handleSubmit}
                disabled={!canSubmit}
                className="w-full flex items-center justify-center gap-3 rounded-2xl font-bold transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed"
                style={{
                  padding: "20px 28px",
                  fontSize: "1.1rem",
                  background: canSubmit ? "linear-gradient(135deg, rgba(212,160,23,0.2), rgba(124,58,237,0.2))" : "rgba(212,160,23,0.04)",
                  border: `1px solid ${canSubmit ? "rgba(212,160,23,0.55)" : "rgba(212,160,23,0.1)"}`,
                  color: canSubmit ? "#d4a017" : "#64748b",
                  fontFamily: MONO,
                  boxShadow: canSubmit ? "0 0 32px rgba(212,160,23,0.2)" : "none",
                  letterSpacing: "0.05em",
                }}
              >
                <Send size={18} />
                Lock in my guess
              </button>
            </div>
          )}
        </div>
      </div>

      {phase === "results" && currentRoundScore && (
        <RoundResults
          score={currentRoundScore}
          round={round}
          totalRounds={TOTAL_ROUNDS}
          cumulativeUserScore={userTotal}
          cumulativeModelScore={modelTotal}
          onNext={handleNext}
          isLastRound={round >= TOTAL_ROUNDS}
        />
      )}
    </div>
  );
}

