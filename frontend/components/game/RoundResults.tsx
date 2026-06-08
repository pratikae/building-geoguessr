"use client";

import { useEffect, useState } from "react";
import { RoundScore } from "@/lib/types";
import { formatDistance, formatScore } from "@/lib/gameLogic";
import { Trophy, Target, Cpu, ChevronRight } from "lucide-react";
import GlobeWrapper from "@/components/globe/GlobeWrapper";
import { GlobePin, GlobeArc } from "@/lib/types";

interface RoundResultsProps {
  score: RoundScore;
  round: number;
  totalRounds: number;
  cumulativeUserScore: number;
  cumulativeModelScore: number;
  onNext: () => void;
  isLastRound: boolean;
}

export default function RoundResults({
  score,
  round,
  totalRounds,
  cumulativeUserScore,
  cumulativeModelScore,
  onNext,
  isLastRound,
}: RoundResultsProps) {
  const [showArcs, setShowArcs] = useState(false);
  const [revealed, setRevealed] = useState(false);

  useEffect(() => {
    const t1 = setTimeout(() => setRevealed(true), 300);
    const t2 = setTimeout(() => setShowArcs(true), 900);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, []);

  const userWon = score.userGuess.score >= score.modelGuess.score;

  const pins: GlobePin[] = [
    {
      lat: score.userGuess.coordinates.lat,
      lng: score.userGuess.coordinates.lng,
      color: "#3b82f6",
      label: `Your guess: ${score.userGuess.country}`,
      size: 0.6,
    },
    {
      lat: score.modelGuess.coordinates.lat,
      lng: score.modelGuess.coordinates.lng,
      color: "#7c3aed",
      label: `Model guess: ${score.modelGuess.country}`,
      size: 0.6,
    },
    {
      lat: score.trueCoordinates.lat,
      lng: score.trueCoordinates.lng,
      color: "#10b981",
      label: `Actual: ${score.trueCountry}`,
      size: 0.8,
    },
  ];

  const arcs: GlobeArc[] = showArcs
    ? [
        {
          startLat: score.userGuess.coordinates.lat,
          startLng: score.userGuess.coordinates.lng,
          endLat: score.modelGuess.coordinates.lat,
          endLng: score.modelGuess.coordinates.lng,
          color: ["#3b82f6", "#7c3aed"],
        },
      ]
    : [];

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center p-4"
      style={{ background: "rgba(5, 5, 9, 0.95)" }}
    >
      <div className="w-full max-w-5xl grid grid-cols-5 gap-4 h-[85vh]">
        {/* Globe panel */}
        <div
          className="col-span-3 rounded-xl overflow-hidden panel relative"
          style={{ minHeight: 0 }}
        >
          <div
            className="absolute top-4 left-4 z-10 px-3 py-1.5 rounded-lg"
            style={{
              background: "rgba(13, 13, 24, 0.85)",
              border: "1px solid rgba(0, 212, 255, 0.2)",
              color: "#64748b",
              fontFamily: "var(--font-geist-mono)",
              fontSize: "0.93rem",
              letterSpacing: "0.12em",
            }}
          >
            ROUND {round} / {totalRounds}
          </div>

          {/* Legend */}
          <div
            className="absolute bottom-3 left-3 z-10 flex flex-col gap-1.5 px-3 py-2 rounded-lg"
            style={{ background: "rgba(13, 13, 24, 0.9)", border: "1px solid rgba(0, 212, 255, 0.15)" }}
          >
            {[
              { color: "#3b82f6", label: "Your guess" },
              { color: "#7c3aed", label: "Bob's guess" },
              { color: "#10b981", label: "Actual location" },
            ].map(({ color, label }) => (
              <div key={label} className="flex items-center gap-2">
                <div className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
                <span
                  className="text-sm"
                  style={{ color: "#64748b", fontFamily: "var(--font-geist-mono)" }}
                >
                  {label}
                </span>
              </div>
            ))}
          </div>

          <GlobeWrapper
            pins={pins}
            arcs={arcs}
            interactive
            focusLat={score.trueCoordinates.lat}
            focusLng={score.trueCoordinates.lng}
            altitude={3}
          />
        </div>

        {/* Score panel */}
        <div
          className="col-span-2 flex flex-col gap-3 rounded-xl p-5 panel"
          style={{ overflow: "hidden" }}
        >
          {/* Header */}
          <div>
            <div
              style={{ color: "#64748b", fontFamily: "var(--font-geist-mono)", fontSize: "0.88rem", letterSpacing: "0.15em", marginBottom: 4 }}
            >
              ROUND {round} RESULTS
            </div>
            <div
              className={`text-xl font-bold transition-all duration-500 ${revealed ? "opacity-100" : "opacity-0"}`}
              style={{ color: userWon ? "#10b981" : "#7c3aed" }}
            >
              {userWon ? "YOU WIN THIS ROUND" : "BOB WINS THIS ROUND"}
            </div>
          </div>

          {/* Building name */}
          <div
            className="text-sm rounded-lg px-3 py-2"
            style={{ background: "rgba(0, 212, 255, 0.05)", border: "1px solid rgba(0, 212, 255, 0.1)", color: "#e2e8f0" }}
          >
            <span style={{ color: "#64748b", fontFamily: "var(--font-geist-mono)", fontSize: "0.88rem" }}>
              BUILDING:{" "}
            </span>
            {score.buildingName}
            <br />
            <span style={{ color: "#64748b", fontFamily: "var(--font-geist-mono)", fontSize: "0.88rem" }}>
              ACTUAL LOCATION:{" "}
            </span>
            <span style={{ color: "#10b981" }}>{score.trueCountry}</span>
          </div>

          {/* Comparison cards */}
          <div className="flex-1 flex flex-col gap-2">
            <ScoreCard
              icon={<Target size={14} />}
              label="YOUR GUESS"
              country={score.userGuess.country}
              distance={score.userGuess.distanceKm}
              score={score.userGuess.score}
              color="#3b82f6"
              winner={userWon}
              revealed={revealed}
            />
            <ScoreCard
              icon={<Cpu size={14} />}
              label="BOB'S GUESS"
              country={score.modelGuess.country}
              distance={score.modelGuess.distanceKm}
              score={score.modelGuess.score}
              color="#7c3aed"
              winner={!userWon}
              revealed={revealed}
            />
          </div>

          {/* Cumulative scores */}
          <div
            className="rounded-lg p-3"
            style={{ background: "#13131f", border: "1px solid rgba(0, 212, 255, 0.1)" }}
          >
            <div
              style={{ color: "#64748b", fontFamily: "var(--font-geist-mono)", fontSize: "0.88rem", letterSpacing: "0.15em", marginBottom: 8 }}
            >
              RUNNING TOTAL
            </div>
            <div className="flex justify-between items-center">
              <div>
                <div className="text-sm" style={{ color: "#3b82f6", fontFamily: "var(--font-geist-mono)" }}>YOU</div>
                <div className="text-xl font-bold" style={{ color: "#e2e8f0" }}>
                  {formatScore(cumulativeUserScore)}
                </div>
              </div>
              <div className="text-2xl" style={{ color: "#64748b" }}>vs</div>
              <div className="text-right">
                <div className="text-sm" style={{ color: "#7c3aed", fontFamily: "var(--font-geist-mono)" }}>BOB</div>
                <div className="text-xl font-bold" style={{ color: "#e2e8f0" }}>
                  {formatScore(cumulativeModelScore)}
                </div>
              </div>
            </div>
          </div>

          {/* Next button */}
          <button
            onClick={onNext}
            className="w-full flex items-center justify-center gap-2 py-3 rounded-lg font-semibold text-sm transition-all"
            style={{
              background: "linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(124, 58, 237, 0.15))",
              border: "1px solid rgba(0, 212, 255, 0.4)",
              color: "#00d4ff",
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLElement).style.background =
                "linear-gradient(135deg, rgba(0, 212, 255, 0.25), rgba(124, 58, 237, 0.25))";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLElement).style.background =
                "linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(124, 58, 237, 0.15))";
            }}
          >
            {isLastRound ? (
              <>
                <Trophy size={16} /> VIEW FINAL RESULTS
              </>
            ) : (
              <>
                NEXT ROUND <ChevronRight size={16} />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

function ScoreCard({
  icon,
  label,
  country,
  distance,
  score,
  color,
  winner,
  revealed,
}: {
  icon: React.ReactNode;
  label: string;
  country: string;
  distance: number;
  score: number;
  color: string;
  winner: boolean;
  revealed: boolean;
}) {
  return (
    <div
      className={`rounded-lg p-3 transition-all duration-500 ${revealed ? "opacity-100" : "opacity-0"}`}
      style={{
        background: winner ? `rgba(${hexToRgb(color)}, 0.08)` : "rgba(0, 0, 0, 0.2)",
        border: `1px solid ${winner ? color : "rgba(0, 212, 255, 0.08)"}`,
        boxShadow: winner ? `0 0 16px rgba(${hexToRgb(color)}, 0.15)` : "none",
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5" style={{ color }}>
          {icon}
          <span className="text-xs font-mono tracking-widest" style={{ fontFamily: "var(--font-geist-mono)" }}>
            {label}
          </span>
          {winner && (
            <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: color, color: "#fff", fontSize: "0.6rem" }}>
              WINNER
            </span>
          )}
        </div>
      </div>
      <div className="text-sm font-medium mb-1" style={{ color: "#e2e8f0" }}>
        {country}
      </div>
      <div className="flex justify-between text-xs" style={{ fontFamily: "var(--font-geist-mono)" }}>
        <span style={{ color: "#64748b" }}>
          Distance:{" "}
          <span style={{ color: "#e2e8f0" }}>{formatDistance(distance)}</span>
        </span>
        <span style={{ color }}>+{formatScore(score)} pts</span>
      </div>
    </div>
  );
}

function hexToRgb(hex: string): string {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) return "0, 212, 255";
  return `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}`;
}

