"use client";

import Link from "next/link";
import { ChevronLeft, Trophy, Cpu } from "lucide-react";

interface GameHeaderProps {
  round: number;
  totalRounds: number;
  userScore: number;
  modelScore: number;
}

export default function GameHeader({ round, totalRounds, userScore, modelScore }: GameHeaderProps) {
  return (
    <header
      className="flex items-center justify-between px-4 py-2.5 flex-shrink-0"
      style={{
        background: "rgba(13, 13, 24, 0.9)",
        borderBottom: "1px solid rgba(0, 212, 255, 0.1)",
        backdropFilter: "blur(8px)",
      }}
    >
      {/* Left: back + logo */}
      <div className="flex items-center gap-3">
        <Link
          href="/"
          className="flex items-center gap-1 text-xs transition-colors"
          style={{ color: "#64748b", fontFamily: "var(--font-geist-mono)" }}
        >
          <ChevronLeft size={14} />
          EXIT
        </Link>
        <div
          className="h-4 w-px"
          style={{ background: "rgba(0, 212, 255, 0.2)" }}
        />
        <span
          className="text-sm font-bold tracking-widest gradient-text"
          style={{ fontFamily: "var(--font-geist-mono)" }}
        >
          ARCHLOC
        </span>
      </div>

      {/* Center: round indicator */}
      <div className="flex items-center gap-2">
        {Array.from({ length: totalRounds }, (_, i) => (
          <div
            key={i}
            className="w-6 h-1.5 rounded-full transition-colors duration-300"
            style={{
              background:
                i < round - 1
                  ? "#00d4ff"
                  : i === round - 1
                  ? "rgba(0, 212, 255, 0.6)"
                  : "rgba(0, 212, 255, 0.1)",
              boxShadow: i === round - 1 ? "0 0 6px #00d4ff" : "none",
            }}
          />
        ))}
        <span
          className="ml-1 text-xs"
          style={{ color: "#64748b", fontFamily: "var(--font-geist-mono)" }}
        >
          {round}/{totalRounds}
        </span>
      </div>

      {/* Right: scores */}
      <div className="flex items-center gap-4">
        <ScorePill
          icon={<Trophy size={12} />}
          label="YOU"
          score={userScore}
          color="#3b82f6"
        />
        <div
          className="h-4 w-px"
          style={{ background: "rgba(0, 212, 255, 0.1)" }}
        />
        <ScorePill
          icon={<Cpu size={12} />}
          label="MODEL"
          score={modelScore}
          color="#7c3aed"
        />
      </div>
    </header>
  );
}

function ScorePill({
  icon,
  label,
  score,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  score: number;
  color: string;
}) {
  return (
    <div className="flex items-center gap-1.5">
      <span style={{ color }}>{icon}</span>
      <span
        className="text-xs"
        style={{ color: "#64748b", fontFamily: "var(--font-geist-mono)" }}
      >
        {label}
      </span>
      <span
        className="text-sm font-bold tabular-nums"
        style={{ color: "#e2e8f0", fontFamily: "var(--font-geist-mono)" }}
      >
        {score.toLocaleString()}
      </span>
    </div>
  );
}
